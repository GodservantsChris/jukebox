import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from jukebox.transformer.ops import filter_logits
from jukebox.transformer.transformer import Transformer
from jukebox.utils.logger import get_range
from jukebox.utils.torch_utils import empty_cache

def get_normal(*shape, std=0.01):
    w = t.empty(shape)
    nn.init.normal_(w, std=std)
    return w

def roll(x, n):
    return t.cat((x[:, -n:], x[:, :-n]), dim=1)

def split_chunks(length, chunk_size):
    n_passes = (length + chunk_size - 1) // chunk_size
    chunk_sizes = [*[chunk_size] * (n_passes - 1), (length - 1) % chunk_size + 1]
    assert sum(chunk_sizes) == length
    return chunk_sizes

class PositionEmbedding(nn.Module):
    def __init__(self, input_shape, width, init_scale=1.0, pos_init=False):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.pos_init = pos_init
        if pos_init:
            self.register_buffer('pos', t.tensor(get_pos_idx(input_shape)).long())
            self._pos_embs = nn.ModuleList()
            for i in range(len(input_shape)):
                emb = nn.Embedding(input_shape[i], width)
                nn.init.normal_(emb.weight, std=0.02)
                self._pos_embs.append(emb)
        else:
            self.pos_emb = nn.Parameter(get_normal(input_dims, width, std=0.01 * init_scale))

    def forward(self):
        if self.pos_init:
            pos_emb = sum([self._pos_embs[i](self.pos[:,i]) for i in range(len(self.input_shape))])
        else:
            pos_emb = self.pos_emb
        return pos_emb

class ConditionalAutoregressive2D(nn.Module):
    def __init__(self, input_shape, bins,
                 width=128, depth=2, heads=1,
                 attn_dropout=0.0, resid_dropout=0.0, emb_dropout=0.0, mask=True,
                 zero_out=False, init_scale=1.0, res_scale=False, pos_init=False,
                 m_attn=0.25, m_mlp=1,
                 checkpoint_res=0, checkpoint_attn=0, checkpoint_mlp=0,
                 attn_order=0, blocks=None, spread=None, x_cond=False, y_cond=False,
                 encoder_dims=0, only_encode=False, merged_decoder=False, prime_len=None):
        super().__init__()
        self.input_shape = input_shape
        self.input_dims = input_dims = np.prod(input_shape)
        self.encoder_dims = encoder_dims
        self.bins = bins
        self.width = width
        self.depth = depth

        self.x_emb = nn.Embedding(bins, width)
        nn.init.normal_(self.x_emb.weight, std=0.02 * init_scale)
        self.x_emb_dropout = nn.Dropout(emb_dropout)
        self.y_cond = y_cond
        self.x_cond = x_cond
        if not y_cond:
            self.start_token = nn.Parameter(get_normal(1, width, std=0.01 * init_scale))

        self.pos_emb = PositionEmbedding(input_shape=input_shape, width=width, init_scale=init_scale, pos_init=pos_init)
        self.pos_emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(n_in=width, n_ctx=input_dims, n_head=heads, n_depth=depth,
                                       attn_dropout=attn_dropout, resid_dropout=resid_dropout,
                                       afn='quick_gelu', scale=True, mask=mask,
                                       zero_out=zero_out, init_scale=init_scale, res_scale=res_scale,
                                       m_attn=m_attn, m_mlp=m_mlp,
                                       checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp, checkpoint_res=checkpoint_res,
                                       attn_order=attn_order, blocks=blocks, spread=spread,
                                       encoder_dims=encoder_dims, prime_len=prime_len)

        self.only_encode = only_encode
        self.prime_len = prime_len
        if merged_decoder:
            # Merged piped model uses this setup
            self.add_cond_after_transformer = False
            self.share_x_emb_x_out = False
        else:
            self.add_cond_after_transformer = True
            self.share_x_emb_x_out = True

        if not only_encode:
            self.x_out = nn.Linear(width, bins, bias=False)
            if self.share_x_emb_x_out:
                self.x_out.weight = self.x_emb.weight
            self.loss = t.nn.CrossEntropyLoss()

    def device(self):
        try:
            return next(self.parameters()).device            
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                device =f"cpu"
                if t.cuda.is_available() : device=f"cuda"
                return device
            
    def preprocess(self, x):
        # Input: x is NHWC and uint8. Converted to NL and long
        # Can include stuff like bitpacking, reordering here.
        N = x.shape[0]
        return x.view(N, -1).long()

    def postprocess(self, x, sample_tokens=None):
        # Convert back from NL and long to NHWC
        N = x.shape[0]
        assert (0 <= x).all() and (x < self.bins).all()
        if sample_tokens is None or sample_tokens==self.input_dims:
            return x.view(N, *self.input_shape)
        else:
            return x.view(N, -1)

    def forward(self, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, loss_full=False,
                encode=False, get_preds=False, get_acts=False, get_sep_loss=False):
        # Preprocess.
        with t.no_grad():
            x = self.preprocess(x)

        N, D = x.shape
        dtype_expected = t.int64
        if t.cuda.is_available() : dtype_expected = t.cuda.LongTensor
        assert (x.dtype == dtype_expected),  f"Expected dtype {dtype_expected}, got {x.dtype}"
        assert (0 <= x).all() and (x < self.bins).all()

        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"{x_cond.shape} != {(N, D, self.width)} nor {(N, 1, self.width)}. Did you pass the correct --sample_length?"
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), device=x.device, dtype=t.float)

        x_t = x # Target
        x = self.x_emb(x) # X emb
        x = roll(x, 1) # Shift by 1, and fill in start token
        if self.y_cond:
            x[:,0] = y_cond.view(N, self.width)
        else:
            x[:,0] = self.start_token

        x = self.x_emb_dropout(x) + self.pos_emb_dropout(self.pos_emb()) + x_cond # Pos emb and dropout

        x = self.transformer(x, encoder_kv=encoder_kv, fp16=fp16) # Transformer
        if self.add_cond_after_transformer: # Piped doesnt add x_cond
            x = x + x_cond

        acts = x
        if self.only_encode:
            return x
        x = self.x_out(x) # Predictions

        if get_sep_loss:
            assert self.prime_len is not None
            x_prime = x[:, :self.prime_len].reshape(-1, self.bins)
            x_gen = x[:, self.prime_len:].reshape(-1, self.bins)

            prime_loss = F.cross_entropy(x_prime, x_t[:, :self.prime_len].reshape(-1)) / np.log(2.)
            gen_loss = F.cross_entropy(x_gen, x_t[:, self.prime_len:].reshape(-1)) / np.log(2.)

            loss = (prime_loss, gen_loss) # Note order! Prime is first
        else:
            loss = F.cross_entropy(x.view(-1, self.bins), x_t.view(-1)) / np.log(2.)  # Loss

        if get_preds:
            return loss, x
        elif get_acts:
            return loss, acts
        else:
            return loss, None

    def get_emb(self, sample_t, n_samples, x, x_cond, y_cond):
        N, D = n_samples, self.input_dims
        if sample_t == 0:
            # Fill in start token
            x = t.empty(n_samples, 1, self.width).to(self.device())
            if self.y_cond:
                x[:, 0] = y_cond.view(N, self.width)
            else:
                x[:, 0] = self.start_token
        else:
            dtype_expected = t.int64
            if t.cuda.is_available() : dtype_expected = t.cuda.LongTensor
            assert (x.dtype == dtype_expected),  f"Expected dtype {dtype_expected}, got {x.dtype}"
            assert (0 <= x).all() and (x < self.bins).all()
            x = self.x_emb(x)
        assert x.shape == (n_samples, 1, self.width)
        if x_cond.shape == (N, D, self.width):
            cond = x_cond[:, sample_t:sample_t + 1, :]
        else:
            cond = x_cond
        x = x + self.pos_emb()[sample_t:sample_t + 1] + cond  # Pos emb, dropout is identity at eval time
        assert x.shape == (n_samples, 1, self.width)
        return x, cond

    def sample(self, n_samples, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0, top_p=0.0,
               get_preds=False, sample_tokens=None):
        assert self.training == False

        if sample_tokens is None: sample_tokens=self.input_dims
        N, D = n_samples, self.input_dims
        if self.y_cond:
            assert y_cond is not None
            assert y_cond.shape == (N, 1, self.width)
        else:
            assert y_cond is None

        if self.x_cond:
            assert x_cond is not None
            assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})"
        else:
            assert x_cond is None
            x_cond = t.zeros((N, 1, self.width), dtype=t.float).to(self.device())

        with t.no_grad():
            xs, x = [], None
            if get_preds:
                preds = []
            for sample_t in get_range(range(0, sample_tokens)):
                x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                self.transformer.check_cache(n_samples, sample_t, fp16)
                x = self.transformer(x, encoder_kv=encoder_kv, sample=True, fp16=fp16) # Transformer
                if self.add_cond_after_transformer:
                    x = x + cond
                assert x.shape == (n_samples, 1, self.width)
                x = self.x_out(x) # Predictions
                if get_preds:
                    preds.append(x.clone())
                # Adjust logits
                x = x / temp
                x = filter_logits(x, top_k=top_k, top_p=top_p)
                x = t.distributions.Categorical(logits=x).sample() # Sample and replace x
                assert x.shape == (n_samples, 1)
                xs.append(x.clone())

            del x
            self.transformer.del_cache()

            x = t.cat(xs, dim=1)
            if get_preds:
                preds = t.cat(preds, dim=1)
            x = self.postprocess(x, sample_tokens)
        if get_preds:
            return x, preds
        else:
            return x

    def primed_sample(self, n_samples, x, x_cond=None, y_cond=None, encoder_kv=None, fp16=False, temp=1.0, top_k=0,
                      top_p=0.0, get_preds=False, chunk_size=None, sample_tokens=None):
        emsgContext = f"autoregressive.primed_sample()"
        emsgOperation = f""
        try: 
            emsgOperation = f"asserting if self.training is False" 
            assert self.training == False, f"self.training != False"
            if sample_tokens is None: 
                emsgOperation = f"setting sample_tokens from self.input_dims"
                sample_tokens=self.input_dims
            # Preprocess.
            with t.no_grad():
                emsgOperation = f"calling preprocess() to set x"
                x = self.preprocess(x)
            emsgOperation = f"setting dtype_expected"
            dtype_expected = t.int64
            if t.cuda.is_available() : dtype_expected = t.cuda.LongTensor
            emsgOperation = f"asserting x.dtype is as expected"
            assert (x.dtype == dtype_expected),  f"Expected dtype {dtype_expected}, got {x.dtype}"
            emsgOperation = f"asserting to validate x"
            assert (0 <= x).all() and (x < self.bins).all()
            emsgOperation = f"asserting to validate x.shape[0]"
            assert x.shape[0] == n_samples, f"x.shape[0] != n_samples"
            emsgOperation = f"splitting x to create xs"
            xs = t.split(x, 1, dim=1)
            emsgOperation = f"re-setting xs as a list"
            xs = list(xs)
            emsgOperation = f"asserting to validate xs"
            assert len(xs) < sample_tokens, f"len(xs) >= sample_tokens"

            N, D = n_samples, self.input_dims
            if self.y_cond:
                emsgOperation = f"asserting to validate y_cond is not None when self.y_cond is defined"
                assert y_cond is not None, f"y_cond is None"
                emsgOperation = f"asserting to validate y_cond.shape when self.y_cond is defined"
                assert y_cond.shape == (N, 1, self.width), f"y_cond.shape != (N, 1, self.width)"
            else:
                emsgOperation = f"asserting to validate y_cond is None when self.y_cond is not defined"
                assert y_cond is None, f"y_cond is not None"

            if self.x_cond:
                emsgOperation = f"asserting to validate x_cond is not None when self.x_cond is defined"
                assert x_cond is not None, f"x_cond is None"
                emsgOperation = f"asserting to validate x_cond.shape when self.x_cond is defined"
                assert x_cond.shape == (N, D, self.width) or x_cond.shape == (N, 1, self.width), f"Got {x_cond.shape}, expected ({N}, {D}/{1}, {self.width})"
            else:
                emsgOperation = f"asserting to validate x_cond is None when self.x_cond is not defined"
                assert x_cond is None, f"x_cond is not None"
                emsgOperation = f"calling t.zeros() to set x_cond when self.x_cond is not defined"
                x_cond = t.zeros((N, 1, self.width), dtype=t.float).to(self.device())

            emsgOperation = f"defining scope as t.no_grad()"
            with t.no_grad():
                if get_preds:
                    preds = []

                # Fill up key/value cache for past context by runing forward pass.
                # We do so in chunks instead of doing the whole past in one forward pass to reduce max memory usage.
                if chunk_size is None:
                    emsgOperation = f"calling len(xs) to set chunk_size"
                    chunk_size = len(xs)
                #assert len(xs) % chunk_size == 0, f'expected {len(xs)} to be divisible by {chunk_size}'
                emsgOperation = f"calling split_chunks() to set chunk_sizes"
                chunk_sizes = split_chunks(len(xs), chunk_size)
                emsgOperation = f"setting some local variables"
                x_primes = []
                start = 0
                x = None
                emsgOperation = f"iterating the return from get_range()"
                for current_chunk_size in get_range(chunk_sizes):
                    emsgOperation = f"iterating the range from start to start + chunk_size when current_chunk_size = " + str(current_chunk_size)
                    xs_prime, conds_prime = [], []
                    for sample_t in range(start, start + current_chunk_size):
                        emsgOperation = f"calling get_emb() when current_chunk_size = " + str(current_chunk_size) + f" and sample_t = " + str(sample_t)
                        x_prime, cond_prime = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                        emsgOperation = f"getting xs[sample_t] when current_chunk_size = " + str(current_chunk_size) + f" and sample_t = " + str(sample_t)
                        x = xs[sample_t]
                        emsgOperation = f"calling xs_prime.append() when current_chunk_size = " + str(current_chunk_size) + f" and sample_t = " + str(sample_t)
                        xs_prime.append(x_prime)
                        emsgOperation = f"calling conds_priime.append() when current_chunk_size = " + str(current_chunk_size) + f" and sample_t = " + str(sample_t)
                        conds_prime.append(cond_prime)
                    emsgOperation = f"re-calculating start when current_chunk_size = " + str(current_chunk_size)
                    start = start + current_chunk_size

                    emsgOperation = f"calling t.cat to set x_prime and cond_prime when current_chunk_size = " + str(current_chunk_size)
                    x_prime, cond_prime = t.cat(xs_prime, dim=1), t.cat(conds_prime, dim=1)
                    emsgOperation = f"asserting to validate x_prime.shape when current_chunk_size = " + str(current_chunk_size)
                    assert x_prime.shape == (n_samples, current_chunk_size, self.width), f"x_prime.shape != (n_samples, current_chunk_size, self.width)"
                    emsgOperation = f"asserting to validate cond_prime.shape when current_chunk_size = " + str(current_chunk_size)
                    assert cond_prime.shape == (n_samples, current_chunk_size, self.width), f"cond_prime.shape != (n_samples, current_chunk_size, self.width)"
                    emsgOperation = f"calling del xs_prime when current_chunk_size = " + str(current_chunk_size)
                    del xs_prime
                    emsgOperation = f"calling del conds_prime when current_chunk_size = " + str(current_chunk_size)
                    del conds_prime
                    if not get_preds:
                        emsgOperation = f"calling del cond_prime when current_chunk_size = " + str(current_chunk_size) + f" and not get_preds"
                        del cond_prime
                    emsgOperation = f"calling self.transformer when current_chunk_size = " + str(current_chunk_size)
                    x_prime = self.transformer(x_prime, encoder_kv=encoder_kv, sample=True, fp16=fp16)

                    if get_preds:
                        if self.add_cond_after_transformer:
                            x_prime = x_prime + cond_prime
                        emsgOperation = f"asserting to validate x_prime.shape when current_chunk_size = " + str(current_chunk_size)
                        assert x_prime.shape == (n_samples, current_chunk_size, self.width), f"x_prime.shape != (n_samples, current_chunk_size, self.width)"
                        emsgOperation = f"calling del cond_prime when current_chunk_size = " + str(current_chunk_size)
                        del cond_prime
                        emsgOperation = f"appending x_prime to x_primes when current_chunk_size = " + str(current_chunk_size)
                        x_primes.append(x_prime)
                    else:
                        emsgOperation = f"calling del x_prime when current_chunk_size = " + str(current_chunk_size) + f" and get_preds is falsey"
                        del x_prime

                if get_preds:
                    emsgOperation = f"calling t.cat to set x_prime when get_preds is defined"
                    x_prime = t.cat(x_primes, dim=1)
                    emsgOperation = f"asserting to validate x_prime.shape when get_preds is defined"
                    assert x_prime.shape == (n_samples, len(xs), self.width), f"x_prime.shape != (n_samples, len(xs), self.width)"
                    emsgOperation = f"calling self.x_out() to set x_prime when get_preds is defined"
                    x_prime = self.x_out(x_prime)  # Predictions
                    emsgOperation = f"appending x_prime to preds when get_preds is defined"
                    preds.append(x_prime)

                emsgOperation = f"emptying cache 1"
                empty_cache()
                emsgOperation = f"calling transformer.check_cache()"
                self.transformer.check_cache(n_samples, len(xs), fp16)

                emsgOperation = f"getting xs[-1] to set x"
                x = xs[-1]
                emsgOperation = f"asserting to validate x.shape"
                assert x.shape == (n_samples, 1), f"x.shape != (n_samples, 1)"
                emsgOperation = f"emptying cache 2"
                empty_cache()
                emsgOperation = f"iterating on get_range() for sample_t"
                for sample_t in get_range(range(len(xs), sample_tokens)):
                    emsgOperation = f"calling get_emb() when sample_t = " + str(sample_t)
                    x, cond = self.get_emb(sample_t, n_samples, x, x_cond, y_cond)
                    emsgOperation = f"calling check_cache() when sample_t = " + str(sample_t)
                    self.transformer.check_cache(n_samples, sample_t, fp16)
                    emsgOperation = f"calling tranformer() when sample_t = " + str(sample_t)
                    x = self.transformer(x, encoder_kv=encoder_kv, sample=True, fp16=fp16) # Transformer
                    if self.add_cond_after_transformer:
                        emsgOperation = f"setting x as x + cond when sample_t = " + str(sample_t)
                        x = x + cond
                    emsgOperation = f"asserting to validate x.shape when sample_t = " + str(sample_t)
                    assert x.shape == (n_samples, 1, self.width), f"x.shape != (n_samples, 1, self.width)"
                    emsgOperation = f"calling x_out() when sample_t = " + str(sample_t)
                    x = self.x_out(x) # Predictions
                    if get_preds:
                        emsgOperation = f"appending x to preds when sample_t = " + str(sample_t) + f" and get_preds is truthy"
                        preds.append(x)
                    # Adjust logits
                    emsgOperation = f"re-calculating x by dividing by temp when sample_t = " + str(sample_t)
                    x = x / temp
                    emsgOperation = f"calling filter_logits() to re-set x when sample_t = " + str(sample_t)
                    x = filter_logits(x, top_k=top_k, top_p=top_p)
                    emsgOperation = f"calling distributions.Categorical() to re-set x when sample_t = " + str(sample_t)
                    x = t.distributions.Categorical(logits=x).sample() # Sample and replace x
                    emsgOperation = f"asserting to validate x.shape when sample_t = " + str(sample_t)
                    assert x.shape == (n_samples, 1), f"x.shape != (n_samples, 1)"
                    emsgOperation = f"appending x.clone() to xs when sample_t = " + str(sample_t)
                    xs.append(x.clone())

                emsgOperation = f"final calling of del x"
                del x
                emsgOperation = f"calling transformer.del_cache()"
                self.transformer.del_cache()

                emsgOperation = f"final calling of t.cat() to re-set x"
                x = t.cat(xs, dim=1)
                if get_preds:
                    emsgOperation = f"final calling of t.cat() to re-set preds"
                    preds = t.cat(preds, dim=1)
                emsgOperation = f"calling postprocess()"
                x = self.postprocess(x, sample_tokens)
            
            if get_preds:
                emsgOperation = f"returning x and preds"
                return x, preds
            else:
                emsgOperation = f"returning x"
                return x

        except AssertionError as e:
            emsg = f'AssertionError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
            raise Exception(emsg)
        except NameError as e:
            emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
            raise Exception(emsg)
        except Exception as e:
            emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
            raise Exception(emsg)

    def check_sample(self, chunk_size):
        bs, l, d = (4, self.input_dims, self.width)
        prime = int(self.input_dims//8*7)
        enc_l = self.encoder_dims
        with t.no_grad():
            y_cond = t.randn(bs, 1, d).to(self.device()) if self.y_cond else None
            x_cond = t.randn(bs, l, d).to(self.device()) if self.x_cond else None
            encoder_kv = t.randn(bs, enc_l, d).to(self.device())

            x, preds_sample = self.sample(bs, x_cond, y_cond, encoder_kv, get_preds=True)
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            x_prime = x.view(bs, -1)[:,:prime]
            # unchunked
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True)
            assert (x.view(bs, -1)[:,:prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"

            # chunked
            x, preds_sample = self.primed_sample(bs, x_prime.clone(), x_cond, y_cond, encoder_kv, get_preds=True, chunk_size=chunk_size)
            assert (x.view(bs, -1)[:,:prime] == x_prime).all(), "Priming samples don't match"
            loss, preds_forw = self.forward(x, x_cond, y_cond, encoder_kv, get_preds=True)
            max_err = t.max(t.abs(preds_sample - preds_forw))
            assert max_err <= 1e-6, f"Max err is {max_err} {[i for i in range(l) if t.max(t.abs(preds_sample - preds_forw)[:, i, :]) > 1e-6]}"


def test_prior(input_shape, encoder_dims, blocks, heads, chunk_size, device):
    bins = 512
    width = 32
    depth = 2
    prime_len = encoder_dims
    for x_cond in [True, False]:
        for y_cond in [True, False]:
            for attn_order in [0,2,6,12]:
                prior = ConditionalAutoregressive2D(input_shape, bins,
                                                    width=width, depth=depth, heads=heads,
                                                    attn_order=attn_order, blocks=blocks,
                                                    x_cond=x_cond, y_cond=y_cond,
                                                    encoder_dims=encoder_dims, prime_len=prime_len).to(device)
                prior.training = False
                prior.check_sample(chunk_size)
                print(f"Checked x_cond: {x_cond}, y_cond: {y_cond}, attn_order: {attn_order}")
            # prior.apply(_convert_mlp_traced)
            # prior.check_sample()
            # print(f"Checked traced x_cond: {x_cond}, y_cond: {y_cond}")


if __name__ == '__main__':
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    setup_dist_from_mpi(port=29600)
    test_cases = [
        ((6144,), 384, 64, 2, 23),
        ((6144,), 384, 64, 2, 8),
        ((8192,), 512, 128, 2, 16),
    ]
    device =f"cpu"
    if t.cuda.is_available() : device=f"cuda"
    for test_case in test_cases:
        test_prior(*test_case)

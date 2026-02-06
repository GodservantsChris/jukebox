import os
import torch as t
import jukebox.utils.dist_adapter as dist

from jukebox.hparams import Hyperparams
from jukebox.data.labels import EmptyLabeller
from jukebox.utils.torch_utils import empty_cache
from jukebox.utils.audio_utils import save_wav, load_audio
from jukebox.make_models import make_model
from jukebox.align import get_alignment
from jukebox.save_html import save_html
from jukebox.utils.sample_utils import split_batch, get_starts
from jukebox.utils.dist_utils import print_once
import fire

# Sample a partial window of length<n_ctx with tokens_to_sample new tokens on level=level
def sample_partial_window(zs, labels, sampling_kwargs, level, prior, tokens_to_sample, hps):
    z = zs[level]
    n_ctx = prior.n_ctx
    current_tokens = z.shape[1]
    if current_tokens < n_ctx - tokens_to_sample:
        sampling_kwargs['sample_tokens'] = current_tokens + tokens_to_sample
        start = 0
    else:
        sampling_kwargs['sample_tokens'] = n_ctx
        start = current_tokens - n_ctx + tokens_to_sample

    return sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)

# Sample a single window of length=n_ctx at position=start on level=level
def sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps):
    emsgContext = f"sample.py.sample_single_window(level=" + str(level) + f";start=" + str(start) + f";)"
    emsgOperation = f""
    try: 
        if zs: 
            emsgOperation = f"setting n samples from hps"      
            n_samples = hps.n_samples
            emsgOperation = f"setting n_ctx from hps"      
            n_ctx = prior.n_ctx
            emsgOperation = f"calculating end"      
            end = start + n_ctx
            # 
            emsgOperation = f"getting z already sampled at current level"      
            z = zs[level][:,start:end]
            emsgOperation = f"determining if sample_tokens is in the sampling kwargs" 
            if 'sample_tokens' in sampling_kwargs:
                # Support sampling a window shorter than n_ctx
                emsgOperation = f"setting sample_tokens' from sampling_kwargs" 
                sample_tokens = sampling_kwargs['sample_tokens']
            else:
                emsgOperation = f"etting sample_tokens' from end-start" 
                sample_tokens = (end - start)
            emsgOperation = f"setting conditioing and new tokens" 
            conditioning_tokens, new_tokens = z.shape[1], sample_tokens - z.shape[1]
            emsgOperation = f"printing message" 
            print_once(f"Sampling {sample_tokens} tokens for [{start},{start+sample_tokens}]. Conditioning on {conditioning_tokens} tokens")
            emsgOperation = f"checking new_tokens" 
            if new_tokens <= 0:
                emsgOperation = f"returning zs when there's nothing new to sample" 
                return zs
            emsgOperation = f"getting z_conds from level above"         
            z_conds = prior.get_z_conds(zs, start, end)
            emsgOperation = f"setting y offset, sample_length and lyrics tokens" 
            y = prior.get_y(labels, start)
            emsgOperation = f"emptying cache" 
            empty_cache()
            emsgOperation = f"setting max_batch_size" 
            max_batch_size = sampling_kwargs['max_batch_size']
            emsgOperation = f"removing max_batch_size from sampling_kwargs"
            del sampling_kwargs['max_batch_size']
            emsgOperation = f"setting z_listy" 
            z_list = split_batch(z, n_samples, max_batch_size)
            emsgOperation = f"setting z_conds_list" 
            z_conds_list = split_batch(z_conds, n_samples, max_batch_size)
            emsgOperation = f"setting y_list" 
            y_list = split_batch(y, n_samples, max_batch_size)
            emsgOperation = f"initializing z_samples" 
            z_samples = []
            emsgOperation = f"iterating lists" 
            for z_i, z_conds_i, y_i in zip(z_list, z_conds_list, y_list):
                # sampling
                emsgOperation = f"calling prior.sample() to set z_samples_i" 
                z_samples_i = prior.sample(n_samples=z_i.shape[0], z=z_i, z_conds=z_conds_i, y=y_i, **sampling_kwargs)
                emsgOperation = f"appending z_samples_i to z_samples" 
                z_samples.append(z_samples_i)
            emsgOperation = f"calling t.cat(...) to set z" 
            z = t.cat(z_samples, dim=0)
            emsgOperation = f"setting sampling_kwargs[max_batch_size]" 
            sampling_kwargs['max_batch_size'] = max_batch_size
            emsgOperation = f"updating z with new sample" 
            z_new = z[:,-new_tokens:]
            emsgOperation = f"calling t.cat() to set zs[level] where level = " + str(level) 
            zs[level] = t.cat([zs[level], z_new], dim=1)
            emsgOperation = f"returning zs in final statement" 
            return zs
        
        else: raise NameError(f"zs is empty.")
    
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)

# Sample total_length tokens at level=level with hop_length=hop_length
def sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):
    emsgContext = f"sample.py.sample_level()"
    emsgOperation = f""
    try:        
        emsgOperation = f"calling print_once"
        print_once(f"Sampling level {level}")
        emsgOperation = f"determining total_lenth condition"
        if total_length >= prior.n_ctx:
            emsgOperation = f"iterating starts returned from get_starts()"
            for start in get_starts(total_length, prior.n_ctx, hop_length):
                emsgOperation = f"calling sample_single_window() to set zs for start = " + str(start)
                zs = sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)
        else:
            emsgOperation = f"calling sample_partial_window()"
            zs = sample_partial_window(zs, labels, sampling_kwargs, level, prior, total_length, hps)
        emsgOperation = f"returning zs"
        return zs
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)

# Sample multiple levels
def _sample(device, zs, labels, sampling_kwargs, priors, sample_levels, hps):
    emsgContext = f"sample.py._sample()"
    emsgOperation = f""
    try:        
        alignments = None
        emsgOperation = f"iterating sample_levels in reverse"
        for level in reversed(sample_levels):
            emsgOperation = f"setting prior at level=" + str(level)
            prior = priors[level]
            #
            emsgOperation = f"calling prior.to(device) at level=" + str(level)
            prior.to(device)
            #
            emsgOperation = f"emptying cache first time at level=" + str(level)
            empty_cache()
            #
            # Set correct total_length, hop_length, labels and sampling_kwargs for level
            emsgOperation = f"asserting sample_length condition at level=" + str(level)
            assert hps.sample_length % prior.raw_to_tokens == 0, f"Expected sample_length {hps.sample_length} to be multiple of {prior.raw_to_tokens}"
            emsgOperation = f"setting total_length at level=" + str(level)
            total_length = hps.sample_length//prior.raw_to_tokens
            emsgOperation = f"setting hop_length at level=" + str(level)
            hop_length = int(hps.hop_fraction[level]*prior.n_ctx)
            #
            # Sample the level
            emsgOperation = f"calling sample_level to set zs at level=" + str(level)
            zs = sample_level(zs, labels[level], sampling_kwargs[level], level, prior, total_length, hop_length, hps)
            #
            emsgOperation = f"calling prior.cpu() at level=" + str(level)
            prior.cpu()
            #
            emsgOperation = f"emptying cache second time at level=" + str(level)
            empty_cache()
            if zs:
                # Decode sample
                emsgOperation = f"decoding prior at level=" + str(level)
                x = prior.decode(zs[level:], start_level=level, bs_chunks=zs[level].shape[0])

                emsgOperation = f"determining if dist.get_world_size() > 1 at level=" + str(level)
                if dist.get_world_size() > 1:
                    emsgOperation = f"setting logdir when dist.get_world_size() > 1 at level=" + str(level)
                    logdir = f"{hps.name}_rank_{dist.get_rank()}/level_{level}"
                else:
                    emsgOperation = f"setting logdir when dist.get_world_size() <= 1 at level=" + str(level)
                    logdir = f"{hps.name}/level_{level}"
                emsgOperation = f"determinint if logdir does not exist at level=" + str(level)
                if not os.path.exists(logdir):
                    emsgOperation = f"making logdir at level=" + str(level)
                    os.makedirs(logdir)
                #
                emsgOperation = f"calliing t.save() at level=" + str(level)
                t.save(dict(zs=zs, labels=labels, sampling_kwargs=sampling_kwargs, x=x), f"{logdir}/data.pth.tar")
                emsgOperation = f"calling save_wav() at level=" + str(level)
                save_wav(logdir, x, hps.sr)
                emsgOperation = f"determining is alignments should be gotten at level=" + str(level)
                if alignments is None and priors[-1] is not None and priors[-1].n_tokens > 0 and not isinstance(priors[-1].labeller, EmptyLabeller):
                    emsgOperation = f"getting alignments at level=" + str(level)
                    alignments = get_alignment(x, zs, labels[-1], priors[-1], sampling_kwargs[-1]['fp16'], hps)
                emsgOperation = f"saving html at level=" + str(level)
                save_html(logdir, x, zs, labels[-1], alignments, hps)
            else:
                raise NameError(f"sz is empty.")
        return zs
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e) 
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)

# Generate ancestral samples given a list of artists and genres
def ancestral_sample(device, labels, sampling_kwargs, priors, hps):
    emsgContext = f"sample.py.ancestral_sample()"
    emsgOperation = f""
    try:        
        emsgOperation = f"setting sample_levels"
        sample_levels = list(range(len(priors)))
        emsgOperation = f"setting zs"
        zs = [t.zeros(hps.n_samples,0,dtype=t.long, device=device) for _ in range(len(priors))]
        emsgOperation = f"calling _sample()"
        zs = _sample(device, zs, labels, sampling_kwargs, priors, sample_levels, hps)
        emsgOperation = f"returning zs"
        return zs
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)

# Continue ancestral sampling from previously saved codes
def continue_sample(device, zs, labels, sampling_kwargs, priors, hps):
    emsgContext = f"sample.py.continue_sample()"
    emsgOperation = f""
    try:        
        sample_levels = list(range(len(priors)))
        zs = _sample(device, zs, labels, sampling_kwargs, priors, sample_levels, hps)
        return zs

    except AssertionError as e:
        emsg = f'AssertionError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)    

# Upsample given already generated upper-level codes
def upsample(device, zs, labels, sampling_kwargs, priors, hps):
    emsgContext = f"sample.py.upsample()"
    emsgOperation = f""
    try:        
        sample_levels = list(range(len(priors) - 1))
        zs = _sample(device, zs, labels, sampling_kwargs, priors, sample_levels, hps)
        return zs

    except AssertionError as e:
        emsg = f'AssertionError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)

# Prompt the model with raw audio input (dimension: NTC) and generate continuations
def primed_sample(device, x, labels, sampling_kwargs, priors, hps):
    emsgContext = f"sample.py.primed_sample()"
    emsgOperation = f""
    try:        
        sample_levels = list(range(len(priors)))
        zs = priors[-1].encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
        zs = _sample(device, zs, labels, sampling_kwargs, priors, sample_levels, hps)
        return zs

    except AssertionError as e:
        emsg = f'AssertionError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)  

# Load `duration` seconds of the given audio files to use as prompts
def load_prompts(audio_files, duration, hps, device):
    print(f"load_prompts(); device = ") + str(device)
    xs = []
    for audio_file in audio_files:
        x = load_audio(audio_file, sr=hps.sr, duration=duration, offset=0.0, mono=True)
        x = x.T # CT -> TC
        xs.append(x)
    while len(xs) < hps.n_samples:
        xs.extend(xs)
    xs = xs[:hps.n_samples]
    x = t.stack([t.from_numpy(x) for x in xs])
    x = x.to(device, non_blocking=True)
    return x

# Load codes from previous sampling run
def load_codes(codes_file, duration, priors, hps):
    device_cur = priors[-1].device()
    print(f"load_codes(); device_cur = ") + str(device_cur)
    data = t.load(codes_file, map_location='cpu')
    zs = [z.to(device_cur) for z in data['zs']]
    assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
    del data
    if duration is not None:
        # Cut off codes to match duration
        top_raw_to_tokens = priors[-1].raw_to_tokens
        assert duration % top_raw_to_tokens == 0, f"Cut-off duration {duration} not an exact multiple of top_raw_to_tokens"
        assert duration//top_raw_to_tokens <= zs[-1].shape[1], f"Cut-off tokens {duration//priors[-1].raw_to_tokens} longer than tokens {zs[-1].shape[1]} in saved codes"
        zs = [z[:,:duration//prior.raw_to_tokens] for z, prior in zip(zs, priors)]
    return zs

# Generate and save samples, alignment, and webpage for visualization.
def save_samples(model, device, hps, sample_hps, metas):
    emsgContext = f"sample.save_samples(model, device, hps, sample_hps)"
    emsgOperation = f""
    try:        
        emsgOperation = f"validating model"
        if model:
            emsgOperation = f"making the model"
            vqvae, priors = make_model(model, device, hps)
            emsgOperation = f"asserting that there is atleast one ctx in get_z_conds. Please choose a longer sample length"
            assert hps.sample_length//priors[-2].raw_to_tokens >= priors[-2].n_ctx, f"Upsampling needs atleast one ctx in get_z_conds. Please choose a longer sample length"
            emsgLoop = f"iterating hps.n_samples"
            while len(metas) < hps.n_samples:
                emsgOperation = emsgLoop + f"; extending metas with itself"
                metas.extend(metas)
            emsgOperation = f"setting metas to last item in metas"
            metas = metas[:hps.n_samples]
            emsgOperation = f"getting labels"
            labels = [prior.labeller.get_batch_labels(metas, device) for prior in priors]
            emsgLoop = f"iterating labels"
            for label in labels:
                emsgOperation = emsgLoop + f"; asserting that the shape of the y label equals hps.n_samples"
                assert label['y'].shape[0] == hps.n_samples
            emsgOperation = f"setting sizes"
            lower_level_chunk_size = 32
            lower_level_max_batch_size = 16
            if model == '1b_lyrics':
                chunk_size = 32
            else:
                chunk_size = 16                
            emsgOperation = f"setting sample_kwargs"
            sampling_kwargs = [dict(temp=0.99, fp16=True, chunk_size=lower_level_chunk_size, max_batch_size=lower_level_max_batch_size),
                            dict(temp=0.99, fp16=True, chunk_size=lower_level_chunk_size, max_batch_size=lower_level_max_batch_size),
                            dict(temp=0.99, fp16=True, chunk_size=chunk_size, max_batch_size=hps.max_batch_size)]
            emsgOperation = f"determining which sample_hps mode is in play"
            if sample_hps.mode == 'ancestral':
                emsgOperation = f"sampling for ancestral mode"
                ancestral_sample(device, labels, sampling_kwargs, priors, hps)
            elif sample_hps.mode in ['continue', 'upsample']:
                emsgOperation = f"asserting that sample_hps.codes_file is set while in continue or upsample mode"
                assert sample_hps.codes_file is not None
                emsgOperation = f"setting top_raw_to_tokens from raw_to_tokens from priors[-1]"
                top_raw_to_tokens = priors[-1].raw_to_tokens
                if sample_hps.prompt_length_in_seconds is not None:
                    emsgOperation = f"setting duration from sample_hps.prompt_length_in_seconds"
                    duration = (int(sample_hps.prompt_length_in_seconds * hps.sr) // top_raw_to_tokens) * top_raw_to_tokens
                else:
                    emsgOperation = f"setting duration to None"
                    duration = None
                emsgOperation = f"loading codes"
                zs = load_codes(sample_hps.codes_file, duration, priors, hps)
                # sample
                if sample_hps.mode == 'continue':
                    emsgOperation = f"sampling for continue mode"
                    continue_sample(device, zs, labels, sampling_kwargs, priors, hps)
                elif sample_hps.mode == 'upsample':
                    emsgOperation = f"sampling for upsample mode"
                    upsample(device, zs, labels, sampling_kwargs, priors, hps)
            elif sample_hps.mode == 'primed':
                emsgOperation = f"asserting that sample_hps.audio_file is set"
                assert sample_hps.audio_file is not None
                emsgOperation = f"asserting that sample_hps.prompt_length_in_sectios is set"
                assert sample_hps.prompt_length_in_seconds is not None
                emsgOperation = f"splitting sample_hps.audio_file"
                audio_files = sample_hps.audio_file.split(',')
                emsgOperation = f"setting top_raw_to_tokens in primed mode"
                top_raw_to_tokens = priors[-1].raw_to_tokens
                emsgOperation = f"calculating duration in primed mode"
                duration = (int(sample_hps.prompt_length_in_seconds * hps.sr) // top_raw_to_tokens) * top_raw_to_tokens
                emsgOperation = f"loading prompts in primed mode"
                x = load_prompts(audio_files, duration, hps, device)
                emsgOperation = f"sampling for primed mode"
                primed_sample(device, x, labels, sampling_kwargs, priors, hps)
            else:
                raise ValueError(f'Unknown sample mode {sample_hps.mode}.')
        else: raise NameError
    except ValueError as e:
        emsg = f'ValueError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except AssertionError as e:
        emsg = f'AssertionError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)

def run(model, backend_to_run='nccl', mode='ancestral', codes_file=None, audio_file=None, prompt_length_in_seconds=None, **kwargs):
    #
    verbose = True
    #
    print(f'mode: ' + str(mode))
    #
    """ arg_lyrics = '''All dressed up to go dreaming
    Now don't tell me I'm wrong
    And what a night to go dreaming
    Mind, if I tag along?

    If I say, I love you, I want you to know
    It's not just because there's moonlight, although
    Moonlight becomes you, moonlight becomes you so''' """
    #
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    emsgContext = f"sample.run()"
    emsgOperation = f""
    try: 
        from datetime import datetime
        # Get the current date and time, format it (e.g., HH:MM:SS) and print it
        now = datetime.now()
        cur_time = now.strftime("%H:%M:%S") 
        print(f'Started: ' + emsgContext + f'; Start time: ' + cur_time)
        # Start the timing for elapsed time
        import time
        start_time = time.perf_counter()
        #     
        emsgOperation = f"validating model input"
        if model:
            emsgOperation = f"setting up distributed devices and getting device from mpi"
            dictSetup = setup_dist_from_mpi(backend = backend_to_run, verbose=verbose)
            if dictSetup:
                rank = dictSetup[0]
                local_rank = dictSetup[1]
                device = dictSetup[2]
                emsgOperation = f"validating device"
                if device:
                    emsgOperation = f"creating Hyperparams from **kwargs (" + str(kwargs) + f")"
                    hps = Hyperparams(**kwargs)
                    if verbose: print(f"hps: ", hps)
                    emsgOperation = f"creating Hyperparams from input args" + f"; rank = " + str(rank)  + f"; local_rank = " + str(local_rank) + f"; device = "+ str(device) + f";"
                    sample_hps = Hyperparams(dict(mode=mode, codes_file=codes_file, audio_file=audio_file, prompt_length_in_seconds=prompt_length_in_seconds))  
                    emsgOperation = f"setting total_length"
                    total_length = hps.total_sample_length_in_seconds * hps.sr
                    emsgOperation = f"setting offset"
                    offset = 0
                    #
                    emsgOperation = f"setting metas"
                    # Set artist/genre/lyrics for your samples here!
                    # We used different label sets in our models, but you can write the human friendly names here and we'll map them under the hood for each model.
                    # For the 5b/5b_lyrics model and the upsamplers, labeller will look up artist and genres in v2 set. (after lowercasing, removing non-alphanumerics and collapsing whitespaces to _).
                    # For the 1b_lyrics top level, labeller will look up artist and genres in v3 set (after lowercasing).
                    metas = [
                            dict(artist=hps.artist,
                                genre=hps.genre,
                                lyrics=hps.lyrics,
                                total_length=total_length,
                                offset=offset,
                                ),
                            ]
                    emsgOperation = f"determining if torch gradient calculations are disabled"
                    with t.no_grad():
                        emsgOperation = f"saving samples when torch gradient calculations are disabled"
                        save_samples(model, device, hps, sample_hps, metas)
                else: raise NameError(f"The device returned is empty.")
            else: raise NameError
        else: raise NameError
    except NameError as e:
        print(f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e))
    except Exception as e:
        print(f'Exception while ' + emsgOperation + f' in ' + emsgContext + f': ' + repr(e))
    finally:
        # Get the current date and time, format it (e.g., HH:MM:SS) and print it
        now = datetime.now()
        cur_time = now.strftime("%H:%M:%S") 
        print(f'Completed: ' + emsgContext + f'; End time: ' + cur_time)
        # Calculate and print the elapsed time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds")

def find_free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def worker_init_process_group(rank, world_size, port):
    import torch.distributed as dist2
    from torch.distributed import TCPStore

    storeTCP = TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=world_size,
        is_master=(rank == 0),
        wait_for_workers=True,
        use_libuv=False
    )

    dist2.init_process_group(
        backend="gloo",
        store=storeTCP,
        rank=rank,
        world_size=world_size
    )

    print(f"Rank {rank} initialization executed.")

def spawn_init_process_group():
    import torch.multiprocessing as mp

    world_size = 2
    port = find_free_port()

    mp.set_start_method("spawn", force=True)
    mp.spawn(worker_init_process_group, args=(world_size,port), nprocs=world_size)

if __name__ == '__main__':
    fire.Fire(run)
    #spawn_init_process_group()

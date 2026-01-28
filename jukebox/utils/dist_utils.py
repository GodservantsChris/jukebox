import os
import socket
from time import sleep
import torch
import jukebox.utils.dist_adapter as dist

def print_once(msg):
    if (not dist.is_available()) or dist.get_rank()==0:
        print(msg)

def print_all(msg):
    if (not dist.is_available()):
        print(msg)
    elif dist.get_rank()%8==0:
        print(f'{dist.get_rank()//8}: {msg}')

def allgather(x):
    xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x)
    xs = torch.cat(xs, dim=0)
    return xs

def allreduce(x, op=dist.ReduceOp.SUM):
    x = torch.tensor(x).float().cuda()
    dist.all_reduce(x, op=op)
    return x.item()

def allgather_lists(xs):
    bs = len(xs)
    total_bs = dist.get_world_size()*len(xs)
    lengths = torch.tensor([len(x) for x in xs], dtype=t.long, device='cuda')
    lengths = allgather(lengths)
    assert lengths.shape == (total_bs,)
    max_length = torch.max(lengths).item()

    xs = torch.tensor([[*x, *[0]*(max_length - len(x))] for x in xs], device='cuda')
    assert xs.shape == (bs, max_length), f'Expected {(bs, max_length)}, got {xs.shape}'
    xs = allgather(xs)
    assert xs.shape == (total_bs,max_length), f'Expected {(total_bs, max_length)}, got {xs.shape}'

    return [xs[i][:lengths[i]].cpu().numpy().tolist() for i in range(total_bs)]

def setup_dist_from_mpi(backend="nccl", verbose=False) -> dict:
    mpi_rank = 0
    local_rank = 0
    device = None
    n_attempts=1
    if backend == "nccl": n_attempts=5
    emsgContext = f"dist_utils.setup_dist_from_mpi(...)"
    emsgOperation = f""
    try: 
        emsgOperation = f"determining if the dist adapter is available"
        if dist.is_available():
            emsgOperation = f"calling _setup_dist_from_mpi"
            mpi_rank, local_rank, device = _setup_dist_from_mpi(backend, n_attempts, verbose)
        else:
            use_cuda = torch.cuda.is_available()
            print(emsgContext + f'; Using cuda {use_cuda}')
            if use_cuda:
                device = torch.device("cuda", local_rank)
                torch.cuda.set_device(local_rank)
            else:
                device = torch.device("cpu")
    except NameError as e:
        print(f'NameError Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e))
    except Exception as e:
        print(f'Exception while ' + emsgOperation + f' in ' + emsgContext + f': ' + repr(e))
    finally:
        print(f'Completed: ' + emsgContext + f'; mpi_rank =  ' + str(mpi_rank) + f'; local_rank =  ' + str(local_rank) + f'; device =  ' + str(device))
        return mpi_rank, local_rank, device

# Distributed device initialization
def _setup_dist_from_mpi(backend: str, n_attempts: int, verbose: bool) -> dict:
    mpi_rank = -1
    local_rank = -1
    device = None
    emsgContext = f"dist_utils._setup_dist_from_mpi(...)"
    emsgOperation = f""
    from mpi4py import MPI  # This must be imported in order to get errors from all ranks to show up
    try:
        emsgOperation = f"validating backend"
        if backend:
            if verbose: print(emsgContext + f": backend = " + backend)
            emsgOperation = f"getting mpi_rank"
            mpi_rank = MPI.COMM_WORLD.Get_rank()
            emsgOperation = f"getting mpi_size"
            mpi_size = MPI.COMM_WORLD.Get_size()
            emsgOperation = f"setting environment variables"
            setup_env(world_size= mpi_size, rank =mpi_rank, verbose=verbose)            
            # Pin this rank to a specific GPU on the node
            local_rank = mpi_rank % 8
            emsgOperation = f"determining if torch.cuda is available"
            if torch.cuda.is_available():
                emsgOperation = f"setting torch.cuda device to local_rank (" + str(local_rank) + ")"
                torch.cuda.set_device(local_rank)
            # There is a race condition when initializing NCCL with a large number of ranks (e.g 500 ranks)
            # We guard against the failure and then retry
            emsgOperation = f"looping to guard against race condition"
            for attempt_idx in range(n_attempts):
                emsgLoop = f""
                try:
                    isInitialized = False
                    emsgLoop = f" initializing process group"
                    dist.init_process_group(backend=backend)
                    emsgLoop = f"determining if the distributed setup is initialized"
                    isInitialized = dist.dist.is_initialized()
                    emsgLoop = f"checking isInitialized"
                    if isInitialized:
                        emsgLoop = f"asserting that dist.get_rank() == mpi_rank (" + str(mpi_rank) + f")"
                        assert dist.get_rank() == mpi_rank
                        emsgLoop = f"setting local rank from mpi_rank % 8"
                        local_rank = mpi_rank % 8                            
                        emsgLoop = f" determining if torch.cuda is available"
                        use_cuda = torch.cuda.is_available()                    
                        if use_cuda:
                            emsgLoop = f"setting device by getting torch.device for cuda with local_rank = " + str(local_rank)
                            device = torch.device("cuda", local_rank) 
                            emsgLoop = f"setting the torch.cuda device to local_rank = " + str(local_rank)
                            torch.cuda.set_device(local_rank)
                        else:
                            emsgLoop = f"setting device by getting torch.device for cpu"
                            device = torch.device("cpu")
                        emsgOperation = f" ending distributed device initialization"
                        break
                    else: raise NameError                                
                except NameError as e:
                    emsg = f'NameError (attempt {attempt_idx + 1} of {n_attempts}) while ' + emsgLoop + f': ' + repr(e)
                    print(emsg)
                    sleep(1 + (0.01 * mpi_rank))  # Sleep to avoid thundering herd
                    pass
                except AttributeError as e:
                    emsg = f'AttributeError (attempt {attempt_idx + 1} of {n_attempts}) while ' + emsgLoop + f': ' + repr(e)
                    print(emsg)
                    sleep(1 + (0.01 * mpi_rank))  # Sleep to avoid thundering herd
                    pass
                except RuntimeError as e:
                    emsg = f'RuntimeError (attempt {attempt_idx + 1} of {n_attempts}) while ' + emsgLoop + f': ' + repr(e)
                    print(emsg)
                    sleep(1 + (0.01 * mpi_rank))  # Sleep to avoid thundering herd
                    pass
                except AssertionError as e:
                    emsg = f'AssertionError (attempt {attempt_idx + 1} of {n_attempts}) while ' + emsgLoop + f': ' + repr(e)
                    print(emsg)
                    sleep(1 + (0.01 * mpi_rank))  # Sleep to avoid thundering herd
                    pass 
                except Exception as e:
                    emsg = f'Exception (attempt {attempt_idx + 1} of {n_attempts}) while ' + emsgLoop + f': ' + repr(e)
                    print(emsg)
                    sleep(1 + (0.01 * mpi_rank))  # Sleep to avoid thundering herd
                    pass                      
        else: raise NameError
    except NameError as e:
        print(f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e))
    except RuntimeError as e:
        print(f'RuntimeError while ' + emsgOperation + f' in ' + emsgContext + f': ' + repr(e) )
    except Exception as e:
        print(f'Exception while ' + emsgOperation + f' in ' + emsgContext + f': ' + repr(e) )
    finally:
        print(f'Completed: ' + emsgContext + f": mpi_rank = " + str(mpi_rank) + f"; local_rank = " + str(local_rank) + f"; device = "+ str(device) + f";")
        return mpi_rank, local_rank, device

def setup_env(world_size=1, rank=0, verbose=False):
    # Detect a valid IP address
    master_addr = get_local_ip()
    os.environ["MASTER_ADDR"] = master_addr
    master_port = get_free_port()
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["USE_LIBUV"] = "0"  # ensure gloo runs without libuv
    os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "2"
    os.environ["NCCL_SOCKET_NTHREADS"] = "8"

def get_local_ip():
    """Detect a valid local IP address for MASTER_ADDR."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # connect to a public DNS server (doesn't send data)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))              # bind to a free ephemeral port
    port = s.getsockname()[1]    # get the port number
    s.close()
    return port

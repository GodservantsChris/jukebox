#from jukebox import sample

def sample_lyrics_1b():
    emsgContext = f"sampling.sample_lyrics_1b"
    emsgOperation = f""
    try:        
        print(f"Sampling...")
    except:
        print(f"Exception in " + emsgContext + f" while " + emsgOperation)
    finally:
        print(f"Done.")   

# Call desired code
sample_lyrics_1b() 


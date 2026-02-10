import os
import json
import numpy as np
from PIL import Image, ImageFilter
import soundfile

def save_html(logdir, x, zs, labels, alignments, hps):
    emsgContext = f"save_html.py.save_html()"
    emsgOperation = f""
    try:  
        emsgOperation = f"getting hps.levels to set level"      
        level = hps.levels - 1 # Top level used
        emsgOperation = f"getting zs[level] to set z"
        z = zs[level]
        emsgOperation = f"getting z.shape[0], z.shape[1] to set bs, total_length"
        bs, total_length = z.shape[0], z.shape[1]
        emsgOperation = f"opening logdir/index.html"
        with open(f'{logdir}/index.html', 'w') as html:
            emsgOperation = f"printing head and title to html"
            print(f"<html><head><title>{logdir}</title></head><body style='font-family: sans-serif; font-size: 1.4em; font-weight: bold; text-align: center; max-width:1024px; width: 100%; margin: auto;'>",
                file=html)
            emsgOperation = f"printing icon link to html"
            print("<link rel='icon' href='data:;base64,iVBORw0KGgo='>", file=html)
            emsgOperation = f"iterating items in bs"
            for item in range(bs):
                emsgOperation = f"creating a data dictionary object when item = " + str(item)
                data = dict(wav=x[item].cpu().numpy(), sr=hps.sr,
                            info=labels['info'][item],
                            total_length=total_length,
                            total_tokens=len(labels['info'][item]['full_tokens']),
                            alignment=alignments[item] if alignments is not None else None)
                emsgOperation = f"definiing item_dir when item = " + str(item)
                item_dir = f'{logdir}/item_{item}'
                emsgOperation = f"calling _save_item_html when item = " + str(item)
                _save_item_html(item_dir, item, item, data)
                emsgOperation = f"printing iframe to html when item = " + str(item)
                print(f"<iframe style='height: 100%; width: 100%;' frameborder='0' scrolling='no' src='item_{item}/index.html'></iframe>", file=html)
            emsgOperation = f"printing closing body and html elements to html"
            print("</body></html>", file=html)

    except AssertionError as e:
        emsg = f'AssertionError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)  

def _save_item_html(item_dir, item_id, item_name, data):
    emsgContext = f"save_html.py._save_item_html()"
    emsgOperation = f""
    try:        
        # replace gs:// with /root/samples/

        # an html for each sample. Main html has a selector to get us id of this?
        emsgOperation = f"make the item dir"
        if not os.path.exists(item_dir):
            os.makedirs(item_dir)
        emsgOperation = f"open item_dir/index.html"
        with open(f'{item_dir}/index.html', 'w') as html:
            emsgOperation = f"print html, head and title to html"
            print(f"<html><head><title>{item_name}</title></head><body style='font-family: sans-serif; font-size: 1.4em; font-weight: bold; text-align: center; max-width:1024px; width: 100%; margin: auto;'>",
                file=html)
            emsgOperation = f"print icon link to html"
            print("<link rel='icon' href='data:;base64,iVBORw0KGgo='>", file=html)
            emsgOperation = f"get data['total_length']"
            total_length = data['total_length']
            emsgOperation = f"get data['total_tokens']"
            total_tokens = data['total_tokens']
            emsgOperation = f"get data['alignment']"
            alignment = data['alignment']
            emsgOperation = f"get data['info']['lyrics']"
            lyrics = data["info"]["lyrics"]
            emsgOperation = f"get data['wav'], data['sr']"
            wav, sr = data['wav'], data['sr']
            emsgOperation = f"get data['info']['genre'], data['info']['artist']"
            genre, artist = data["info"]["genre"], data["info"]["artist"]

            # Strip unused columns
            if alignment is not None:
                emsgOperation = f"assert to validate alignment.shape"
                assert alignment.shape == (total_length, total_tokens)
                emsgOperation = f"assert to validate lyrics"
                assert len(lyrics) == total_tokens, f'Total_tokens: {total_tokens}, Lyrics Len: {len(lyrics)}. Lyrics: {lyrics}'
                emsgOperation = f"get np.max(alignment, axis=0)"
                max_attn_at_token = np.max(alignment, axis=0)
                emsgOperation = f"assert to validate max_attn_at_token"
                assert len(max_attn_at_token) == total_tokens
                emsgOperation = f"iterate total_tokens in reverse"
                for token in reversed(range(total_tokens)):
                    emsgOperation = f"check max_attn_at_token[token] when token = " + str(token)
                    if max_attn_at_token[token] > 0:
                        break
                emsgOperation = f"get alignment[:,:token+1]"
                alignment = alignment[:,:token+1]
                emsgOperation = f"get lyrics[:token+1]"
                lyrics = lyrics[:token+1]
                emsgOperation = f"increment total_tokens"
                total_tokens = token+1

                # Small alignment image
                emsgOperation = f"1st calling of Image.fromarray() using alignment"
                im = Image.fromarray(np.uint8(alignment * 255)).resize((512, 1024)).transpose(Image.ROTATE_90)
                img_src = f'align.png'
                emsgOperation = f"calling im.save()"
                im.save(f'{item_dir}/{img_src}')
                emsgOperation = f"printing image to html"
                print(f"<img id='{img_src}' src='{img_src}' \>", file=html)

                # Smaller alignment json for animation
                emsgOperation = f"setting total_alignment_length"
                total_alignment_length = total_length // 16
                emsgOperation = f"2nd calling of Image.fromarray() using alignment"
                alignment = Image.fromarray(np.uint8(alignment * 255)).resize((total_tokens, total_alignment_length))
                emsgOperation = f"calling alignment.filter()"
                alignment = alignment.filter(ImageFilter.GaussianBlur(radius=1.5))
                emsgOperation = f"calling np.asarray(alignment).tolist()"
                alignment = np.asarray(alignment).tolist()
                align_src = f'align.json'
                emsgOperation = f"opening item_dir/align_src file"
                with open(f'{item_dir}/{align_src}', 'w') as f:
                    emsgOperation = f"calling json.dump() using alignment"
                    json.dump(alignment, f)

            # Audio
            wav_src = f'audio.wav'
            emsgOperation = f"calling soundfile.write()"
            soundfile.write(f'{item_dir}/{wav_src}', wav, samplerate=sr, format='wav')
            emsgOperation = f"printing audio element to html"
            print(f"<audio id='{wav_src}' src='{wav_src}' style='width: 100%;' controls></audio>", file=html)


            # Labels and Lyrics
            emsgOperation = f"printing pre element to html"
            print(f"<pre style='white-space: pre-wrap;'>", end="", file=html)
            emsgOperation = f"printing artist div to html"
            print(f"<div>Artist {artist}, Genre {genre}</div>", file=html)
            emsgOperation = f"1st re-setting of lyrics"
            lyrics = [c for c in lyrics]  # already characters actually
            emsgOperation = f"2nd re-setting of lyrics"
            lyrics = [''] + lyrics[:-1]  # input lyrics are shifted by 1
            emsgOperation = f"iterating lyrics"
            for i, c in enumerate(lyrics):
                emsgOperation = f"printing i (" + str(i) + ") and c (" + str(c) + ") for lyrics"
                print(f"<span id='{item_id}/{i}'>{c}</span>", end="", file=html)
            emsgOperation = f"printing closing of pre element to html"
            print(f"</pre>", file=html)
            emsgOperation = f"opening lyrics.json file"
            with open(f'{item_dir}/lyrics.json', 'w') as f:
                emsgOperation = f"dumping lyrics to lyrics.json file"
                json.dump(lyrics, f)

            if alignment is not None:
                # JS for alignment animation
                emsgOperation = f"printing JS for alignment animation to html"
                print("""<script>
                async function fetchAsync (url) {
                    let response = await fetch(url);
                    let data = await response.json();
                    return data;
                }
        
                var audio = document.getElementById('""" + f'{wav_src}' + """');
                audio.onplay = function () {
                    track = '""" + f'{item_id}' + """'
                    fetchAsync('""" + f'{align_src}' + """')
                    .then(data => animateLyrics(data, track, this))
                    .catch(reason => console.log(reason.message))
                }; 
        
                function animateLyrics(data, track, audio) {
                    var animate = setInterval(function () {
                        var time = Math.floor(audio.currentTime*""" + f'{total_alignment_length}' + """/audio.duration);
                        if (!(time == 0 || time == """ + f'{total_alignment_length}' + """)) {
                            console.log(time);
                            changeColor(data, track, audio, time);
                        }
                        if (audio.paused) {
                            clearInterval(animate);
                        }
                    }, 50);
                }
        
                function changeColor(data, track, audio, time) {
                    colors = data[time]
                    for (i = 0; i < colors.length; i++){
                        character = document.getElementById(track + '/' + i.toString());
                        color = Math.max(230 - 10*colors[i], 0).toString();
                        character.style.color = 'rgb(255,' + color + ',' + color + ')';
                    }
                }
                </script>""", file=html)
            emsgOperation = f"printing closing of body and html to html"
            print("</body></html>", file=html)

    except AssertionError as e:
        emsg = f'AssertionError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except NameError as e:
        emsg = f'NameError while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)
    except Exception as e:
        emsg = f'Exception while ' + emsgOperation + ' in ' + emsgContext + f': ' + repr(e)        
        raise Exception(emsg)


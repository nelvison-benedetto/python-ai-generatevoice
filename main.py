import keyboard
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS   #generate voice from text
import torch
from playsound import playsound  #reproduce audio file
import os
import whisper  #work good with python 3.8>
import pyaudio  #x recording
import wave   #x save audio file in .wav

#TTS multilingual & italians
# tts_models/en/ljspeech/glow-tts  #first downloaded
# tts_models/multilingual/multi-dataset/xtts_v2
# tts_models/it/mai_female/glow-tts
# tts_models/it/mai_female/vits
# tts_models/it/mai_male/glow-tts
# tts_models/it/mai_male/vits

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  #ignora conflitti runtime OpenMP, problema con multilingua

output_file_path = "output_audio.wav"  #name creation audio file
device = "cuda" if torch.cuda.is_available() else "cpu"  #use cpu to elaborate
modelwh = whisper.load_model("medium")  #load model


# Params x Recording
FORMAT = pyaudio.paInt16  #audio format 16bit
CHANNELS = 1  #main, 2 for stereo
RATE = 16000  #heartz
CHUNK = 1024
RECORD_SECONDS = 15
WAVE_OUTPUT_FILENAME = "audio1.mp3"  #"recorded_audio.wav" load this file audio

# Funct to record
def record_audio(output_filename):
    audio = pyaudio.PyAudio()
    # Configura lo stream di input
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    # Registra per il numero di secondi specificato
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording.")
    # Ferma e chiudi lo stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # Salva i dati registrati in un file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("finished func Recording")

def start_whisper():
    audiowh = whisper.load_audio(WAVE_OUTPUT_FILENAME)  #load the audio to whisper
    audiowh = whisper.pad_or_trim(audiowh)  #trunc the silence or add silence to get a stable lenght x whisper
    mel_wh = whisper.log_mel_spectrogram(audiowh).to(modelwh.device)  #convert audio in spectrogramma
    _, probs = modelwh.detect_language(mel_wh)  #use model whisper to get the language
      #probs is dictionary con key= it/en/fr/ectect e value= probabilità associate a ciascuna lingua
    print(f"detected language: {max(probs, key=probs.get)}")  #trova lingua con la probabilita piu alta dal dict probs
    optionswh = whisper.DecodingOptions(fp16=True)  #setting opzioni decod in fp16 piu veloce, se =False usa fp32 piu preciso
    resultwh = whisper.decode(modelwh, mel_wh, optionswh)  #decodifica finale con i settings
    print(resultwh.text)

def go_record(event):
    if event.name == "a":
        print("pressed the A")
        record_audio(WAVE_OUTPUT_FILENAME)
def go_whisper(event):
    if event.name == "s":
        print("pressed the S")
        start_whisper()

keyboard.on_press(go_record)
keyboard.on_press(go_whisper)

print("Press a Button to start...")
keyboard.wait('esc')

#!!VOGLIO RICONOSCIMENTO "START RECORD" & "END RECORD" per definire il file audio wav

#RECORD THE AUDIO FILE
#modelwh = whisper.load_model("medium")

#SPOKE AI
# firstmodel_voice ="tts_models/it/mai_male/glow-tts"  #use fermale english voice
# #model_multilingual = "tts_models/multilingual/multi-dataset/xtts_v2"
# tts = TTS(firstmodel_voice,progress_bar=True).to(device)
# tts.tts_to_file(text="Prova Test in Ita uno due tre quattro", file_path=output_file_path)
# playsound(output_file_path)

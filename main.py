import keyboard
from TTS.api import TTS
import torch
from playsound import playsound
import os
import whisper
import pyaudio
import wave

'''
  premi A  ->  registra 15 secondi
  premi S  ->  Whisper trascrive → TTS legge il testo ad alta voce
  premi Esc ->  esci
'''

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#audio recording params
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 15
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"   # fix: era .mp3 ma wave scrive WAV raw
OUTPUT_TTS_FILE = "output_audio.wav"

#models loaded lazily al primo utilizzo
_whisper_model = None
_tts_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (medium)...")
        _whisper_model = whisper.load_model("medium")
        print("Whisper ready.")
    return _whisper_model

def get_tts_model():
    global _tts_model
    if _tts_model is None:
        print("Loading TTS model...")
        _tts_model = TTS("tts_models/it/mai_female/glow-tts", progress_bar=True).to(device)
        print("TTS ready.")
    return _tts_model

def record_audio(output_filename):
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("Recording...")
        frames = []
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Finished recording.")

        stream.stop_stream()
        stream.close()

        #get_sample_size va chiamato prima di terminate()!
        sample_size = audio.get_sample_size(FORMAT)
        audio.terminate()

        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_size)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        print(f"Audio saved to {output_filename}")

    except OSError as e:
        print(f"[Record Error] Microfono non disponibile: {e}")
    except Exception as e:
        print(f"[Record Error] {e}")

def transcribe_and_speak():
    if not os.path.exists(WAVE_OUTPUT_FILENAME):
        print("Nessun audio trovato. Premi A per registrare prima.")
        return

    try:
        model = get_whisper_model()

        #uso transcribe() invece di decode() manuale!: gestisce audio >30s con segmentazione automatica & fp16 automatico in base al device (fix crash su CPU)
        print("Transcribing...")
        result = model.transcribe(WAVE_OUTPUT_FILENAME)
        text = result["text"].strip()
        lang = result.get("language", "unknown")

        print(f"Language detected: {lang}")
        print(f"Transcribed text: {text}")

        if not text:
            print("Testo vuoto, niente da riprodurre.")
            return

        tts = get_tts_model()
        tts.tts_to_file(text=text, file_path=OUTPUT_TTS_FILE)
        playsound(OUTPUT_TTS_FILE)

    except Exception as e:
        print(f"[Whisper/TTS Error] {e}")

def go_record(event):
    if event.name == "a":
        print("--- [A] Avvio registrazione ---")
        record_audio(WAVE_OUTPUT_FILENAME)

def go_whisper(event):
    if event.name == "s":
        print("--- [S] Avvio trascrizione + TTS ---")
        transcribe_and_speak()

keyboard.on_press(go_record)
keyboard.on_press(go_whisper)

print("=" * 40)
print(" A  = Registra audio (15s)")
print(" S  = Trascrivi + leggi ad alta voce")
print(" Esc = Esci")
print("=" * 40)
keyboard.wait('esc')

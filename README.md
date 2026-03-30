# python-ai-generatevoice

A real-time **Speech-to-Text-to-Speech** pipeline built in Python. The application records microphone input, transcribes it with OpenAI Whisper, and reads it back using a custom Italian Glow-TTS model — all driven by global keyboard shortcuts.

---

## Architecture overview

```
Microphone
    │
    │  PyAudio  (PCM 16-bit mono 16 kHz)
    ▼
recorded_audio.wav
    │
    │  OpenAI Whisper "medium"  (STT)
    ▼
Transcribed text  +  detected language
    │
    │  Coqui TTS  ─  Glow-TTS  (Italian female voice)
    ▼
output_audio.wav
    │
    │  playsound
    ▼
Speaker output
```

Both AI models are **lazy-loaded** on first use: startup is instant, and the ~1 GB memory footprint is allocated only when needed.

---

## Tech stack

| Layer | Library / Model | Version |
|---|---|---|
| Speech-to-Text | `openai-whisper` medium | 20231117 |
| Text-to-Speech | `TTS` (Coqui) — Glow-TTS | 0.22.0 |
| Deep learning | `torch` / `torchaudio` | 2.4.0 |
| Audio I/O | `PyAudio` | 0.2.14 |
| Audio playback | `playsound` | 1.2.2 |
| Keyboard hooks | `keyboard` | 0.13.5 |
| Runtime | Python 3.11+ | — |

GPU acceleration is automatically enabled when CUDA is available; otherwise the pipeline runs on CPU.

---

## Controls

| Key | Action |
|-----|--------|
| `A` | Record 15 seconds of audio from the default microphone |
| `S` | Transcribe the last recording with Whisper → synthesize and play back with TTS |
| `Esc` | Exit the application |

The `keyboard` library registers **global** hotkeys, so the terminal window does not need to be in focus.

---

## Audio pipeline — technical details

### Recording (`record_audio`)

```
PyAudio stream
  format   : paInt16   (16-bit signed PCM)
  channels : 1         (mono)
  rate     : 16 000 Hz (Whisper native rate)
  chunk    : 1 024 frames per read
  duration : 15 seconds  → 234 chunks
```

All chunks are accumulated in memory and written as a standard RIFF/WAV file via the built-in `wave` module. `audio.get_sample_size()` is called **before** `audio.terminate()` to avoid a known PyAudio ordering bug.

### Transcription (`whisper.transcribe`)

`model.transcribe()` is used instead of the lower-level `whisper.decode()` because it:
- handles audio longer than 30 s via automatic segmentation,
- selects fp16/fp32 precision automatically based on the detected device (prevents crashes on CPU-only systems).

The result dict provides both `text` (full transcript) and `language` (auto-detected locale).

### Speech synthesis (`TTS.tts_to_file`)

The TTS model identifier is:

```
tts_models/it/mai_female/glow-tts
```

This resolves to the local checkpoint in `glowtts-female-it/best_model.pth.tar`. The model produces a mel-spectrogram which is converted to a WAV waveform and saved to `output_audio.wav`.

---

## Glow-TTS model — architecture

The model in `glowtts-female-it/` is a **custom-trained Italian female voice** based on the [Glow-TTS](https://arxiv.org/abs/2005.11129) architecture (flow-based generative model).

### Training data

| Property | Value |
|---|---|
| Dataset | Italian LJSpeech variant (`z-uo/female-LJSpeech-italian`) |
| Speaker | Lisa Caputo reading *I Malavoglia* |
| Language | Italian (`it-it`) |

### Model configuration (`config.json`)

**Audio features**

| Parameter | Value |
|---|---|
| Sample rate | 16 000 Hz |
| FFT size | 1 024 |
| Hop length | 256 |
| Mel bins | 80 |
| Mel frequency range | 0 – Nyquist |

**Encoder** — Relative position transformer

| Parameter | Value |
|---|---|
| Layers | 6 |
| Attention heads | 2 |
| FFN hidden units | 768 |
| Kernel size | 3 × 3 |
| Dropout | 0.1 |
| Hidden channels | 192 |

**Decoder** — Normalizing flow

| Parameter | Value |
|---|---|
| Flow blocks | 12 |
| Hidden channels | 192 |
| Duration predictor hidden | 256 |

**Phonemizer** — eSpeak Italian (`it-it`), 66-character set (A–Z, a–z, punctuation, IPA phonemes).

**Optimizer** — RAdam, LR 0.001, weight decay 1e-6, NoamLR scheduler (4 000 warm-up steps), gradient clipping 5.0, batch size 128, trained for 1 000 epochs.

### Available checkpoints

| File | Steps | Notes |
|---|---|---|
| `best_model.pth.tar` | — | Default, used by the app |
| `best_model_6578.pth.tar` | 6 578 | Alternative checkpoint |
| `GOOD_best_model_2530.pth.tar` | 2 530 | Backup |
| `GOOD2_best_model_6578.pth.tar` | 6 578 | Backup variant |
| `checkpoint_10000.pth.tar` | 10 000 | Final training checkpoint |

Each `.pth.tar` file is ~328 MB.

---

## Project structure

```
ownprj_TextSpeechLLAMA/
├── main.py                        # Application entry point
├── glowtts-female-it/
│   ├── config.json                # Glow-TTS model configuration
│   ├── best_model.pth.tar         # Model weights used at runtime (328 MB)
│   ├── best_model_6578.pth.tar    # Alternative checkpoint
│   ├── GOOD_best_model_2530.pth.tar
│   ├── GOOD2_best_model_6578.pth.tar
│   ├── checkpoint_10000.pth.tar
│   ├── trainer_0_log.txt          # Training log
│   └── README.md                  # Hugging Face model card
├── recorded_audio.wav             # Last microphone recording (runtime)
└── output_audio.wav               # Last synthesized speech (runtime)
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nelvison-benedetto/python-ai-generatevoice.git
cd python-ai-generatevoice

# 2. Clone the TTS model weights
git clone https://huggingface.co/z-uo/glowtts-female-it

# 3. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install openai-whisper TTS torch torchaudio pyaudio playsound keyboard

# On Linux, install PortAudio first:
# sudo apt-get install portaudio19-dev
```

> **Windows note:** if PyAudio installation fails, use the unofficial wheel:
> `pip install pipwin && pipwin install pyaudio`

---

## Running the application

```bash
python main.py
```

The terminal will print the detected device (CUDA / CPU) and the keyboard controls. Press `A` to start recording, wait for the 15-second capture to complete, then press `S` to transcribe and hear the synthesised voice.

---

## Using the TTS model standalone

```bash
# Single inference via CLI
tts --text "ciao mondo" \
    --model_path "glowtts-female-it/best_model.pth.tar" \
    --config_path "glowtts-female-it/config.json"

# HTTP server mode (web UI at http://localhost:5002)
tts-server \
    --model_path "glowtts-female-it/best_model.pth.tar" \
    --config_path "glowtts-female-it/config.json"
```

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Any x86-64 | Modern multi-core |
| RAM | 4 GB | 8 GB+ |
| GPU | — (optional) | CUDA-capable, 4 GB VRAM |
| Microphone | Required | — |
| OS | Windows / Linux / macOS | — |

On CPU-only machines the pipeline works correctly but inference (especially Whisper medium) is noticeably slower.

---

## Environment variable

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

Set at startup to suppress a known conflict between the MKL and OpenMP runtime libraries that can occur when PyTorch and certain audio libraries are loaded together in the same process.

---

## Related resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Glow-TTS paper](https://arxiv.org/abs/2005.11129)
- [Hugging Face model card](https://huggingface.co/z-uo/glowtts-female-it)
- [Training script reference](https://github.com/nicolalandro/train_coqui_tts_ita)

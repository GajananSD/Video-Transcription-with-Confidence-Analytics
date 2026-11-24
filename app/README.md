# Video Transcription with Confidence Analytics (`app/`)

**Course project:** CS698B — Conversational Intelligence Platform

Upload a video or audio file and get back a transcript with **who spoke when** and a **calibrated confidence score for every word**. You can correct mistakes, and the system **learns from your corrections**: corrected words become hints for future transcriptions, and confidence scores are re-calibrated.

The project has two parts: a FastAPI backend that runs the speech models, and a Streamlit dashboard for uploading, reviewing, correcting and exporting transcripts.

---

## Contents

1. [Features](#1-features)
2. [Requirements](#2-requirements)
3. [Install system packages (Linux / Windows)](#3-install-system-packages)
4. [Install the app (one time)](#4-install-the-app-one-time)
5. [Configuration (`.env`)](#5-configuration-env)
6. [Running the app](#6-running-the-app)
7. [Using the dashboard](#7-using-the-dashboard)
8. [Stopping the app](#8-stopping-the-app)
9. [Running tests](#9-running-tests)
10. [REST API](#10-rest-api)
11. [Architecture](#11-architecture)
12. [Project layout](#12-project-layout)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Features

- **Media ingestion:** MP4, MOV, WAV, MP3, M4A, AAC and FLAC. Audio is converted to 16 kHz mono with FFmpeg and loudness-normalised. SHA-256 deduplication means re-uploading the same file reuses the existing recording.
- **Speech engine:** voice activity detection, then Faster-Whisper `large-v3` transcription, Wav2Vec2 word alignment, speaker diarization (NeMo MSDD or Pyannote 3.1) and word-to-speaker fusion.
- **Calibrated confidence:** per-word confidence with isotonic calibration. Calibration quality is reported as Brier score, ECE and a reliability diagram.
- **Adaptive learning:** correct a transcript turn's text or speaker. Corrected words are saved as **domain vocabulary** and passed to Whisper as a prompt on later jobs. Every reviewed word becomes a labelled sample that **re-fits the confidence calibrator**. You can then re-transcribe a recording with what was learned.
- **Accessible dashboard:** colourblind-safe palettes, and confidence shown by colour, shape *and* underline. It also has a plain-text mode for screen readers, an audio player, charts with table equivalents, and export to TXT, JSON, SRT and WebVTT.
- **REST API:** versioned `/api/v1` endpoints with Swagger UI at `/docs` and ReDoc at `/redoc`.

---

## 2. Requirements

| Requirement | Notes |
|---|---|
| Python **3.10+** | Tested with 3.14. See [section 3](#3-install-system-packages) for how to install it on your OS. |
| **FFmpeg** on `PATH` | See [section 3](#3-install-system-packages). |
| Disk space | ~3 GB for the Whisper `large-v3` model (downloaded automatically on first run), plus ~2–5 GB for PyTorch. |
| RAM | 8 GB minimum, 16 GB recommended for `large-v3` on CPU. |
| GPU (optional) | An NVIDIA GPU with CUDA is much faster. Everything also works on CPU. |
| Conda (Miniconda/Anaconda) | For the separate **NeMo** environment that produces speaker labels. See [section 5](#5-configuration-env). |
| Hugging Face token (fallback only) | Only needed if NeMo can't be set up. Speaker labels then come from Pyannote. |

---

## 3. Install system packages

Do the part for your operating system, then continue with [section 4](#4-install-the-app-one-time).

### Linux

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git
```

Check that it worked:

```bash
python3 --version     # must be 3.10 or newer
ffmpeg -version
```

### Windows 10 / 11

Use **PowerShell** (not the old Command Prompt) for every Windows command in this README.

**Option A: with `winget`** (built into Windows 10 21H2+ and Windows 11)

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Gyan.FFmpeg
winget install -e --id Git.Git
```

**Option B: manual download**

1. **Python:** download the installer from <https://www.python.org/downloads/windows/>. On the first screen, tick **"Add python.exe to PATH"**, then click **Install Now**.
2. **FFmpeg:** download a "release full" build from <https://www.gyan.dev/ffmpeg/builds/>, unzip it to `C:\ffmpeg`, then add `C:\ffmpeg\bin` to your `PATH` (**Start → "Edit the system environment variables" → Environment Variables → Path → Edit → New**).
3. **Git** (optional, only to clone the repo): <https://git-scm.com/download/win>.

**Close and reopen PowerShell** so the updated `PATH` is picked up, then check:

```powershell
py --version          # must be 3.10 or newer
ffmpeg -version
```

**Allow PowerShell to activate virtual environments** (one time, for your user only):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

> Without this, `.venv\Scripts\Activate.ps1` fails with *"running scripts is disabled on this system"*.

---

## 4. Install the app (one time)

Run all commands from the `app/` directory.

### Linux

```bash
cd Video-Transcription-with-Confidence-Analytics/app

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. Install PyTorch. Pick ONE:
#    a) CPU only (smaller download, no NVIDIA GPU):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#    b) NVIDIA GPU: skip this step. The default Linux wheels already include CUDA.

# 3. Install the app and all its dependencies
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd Video-Transcription-with-Confidence-Analytics\app

# 1. Create and activate a virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Install PyTorch. Pick ONE:
#    a) CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
#    b) NVIDIA GPU (the default Windows wheels are CPU-only, so you must pick a CUDA build):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install the app and all its dependencies
pip install -r requirements.txt
```

> For other CUDA versions, use the selector at <https://pytorch.org/get-started/locally/>.

### About `requirements.txt`

`requirements.txt` simply runs `pip install -e .[dev]`, so the dependency list lives only in `pyproject.toml`. Either command installs the same packages:

```bash
pip install -r requirements.txt
# or
pip install -e ".[dev]"
```

Once the virtual environment is active, use `python` (not `python3` or `py`) on every OS.

---

## 5. Configuration (`.env`)

### Step 1: Create the `.env` file

The app reads its settings from a file named `.env` in the `app/` folder. Create it from the template:

```bash
# Linux
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

`.env` is listed in `.gitignore`; **never commit it or share it**.

Speaker labels ("who spoke when") come from **NeMo MSDD + TitaNet** (Step 2). This needs no Hugging Face account. Only if NeMo can't be set up, use the **Hugging Face / Pyannote fallback** ([below](#fallback-hugging-face--pyannote-diarization)).

### Step 2: Set up NeMo diarization (recommended)

NeMo runs in its **own conda environment** named `cs698b_nemo`, because its dependencies conflict with the app's. Do this once, in a new terminal (the app's `.venv` must **not** be active).

**Linux**

```bash
conda create -n cs698b_nemo python=3.10 -y
conda activate cs698b_nemo

# PyTorch: CPU build (for an NVIDIA GPU use --index-url https://download.pytorch.org/whl/cu124)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install Cython
pip install "nemo_toolkit[asr]==2.7.3"

# Download the models (~150 MB) into ~/.cache/torch/NeMo/
python -c "from nemo.collections.asr.models import EncDecDiarLabelModel, EncDecSpeakerLabelModel; EncDecDiarLabelModel.from_pretrained('diar_msdd_telephonic', map_location='cpu'); EncDecSpeakerLabelModel.from_pretrained('titanet_large', map_location='cpu'); print('NeMo OK')"

conda deactivate
```

**Windows (Anaconda Prompt or PowerShell with conda)**

```powershell
conda create -n cs698b_nemo python=3.10 -y
conda activate cs698b_nemo

# PyTorch: CPU build (for an NVIDIA GPU use --index-url https://download.pytorch.org/whl/cu124)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install Cython
pip install "nemo_toolkit[asr]==2.7.3"

# Download the models (~150 MB) into %USERPROFILE%\.cache\torch\NeMo\
python -c "from nemo.collections.asr.models import EncDecDiarLabelModel, EncDecSpeakerLabelModel; EncDecDiarLabelModel.from_pretrained('diar_msdd_telephonic', map_location='cpu'); EncDecSpeakerLabelModel.from_pretrained('titanet_large', map_location='cpu'); print('NeMo OK')"

conda deactivate
```

> NeMo has limited native Windows support. If `pip install nemo_toolkit` fails, run the whole app inside **WSL2** (Ubuntu) using the Linux steps, or use the [Hugging Face fallback](#fallback-hugging-face--pyannote-diarization).

### Step 3: Point the app at NeMo

The app needs **two paths** in `app/.env`:

| Variable | What it is |
|---|---|
| `NEMO_PYTHON` | The Python program inside the `cs698b_nemo` conda env |
| `NEMO_MSDD_CHECKPOINT` | The downloaded NeMo model file `diar_msdd_telephonic.nemo` |

**3a. Find `NEMO_PYTHON`**

Same commands on Linux and Windows:

```bash
conda activate cs698b_nemo
python -c "import sys; print(sys.executable.replace(chr(92), '/'))"
conda deactivate
```

It prints one path, for example:

```
Linux:   /home/<you>/miniconda3/envs/cs698b_nemo/bin/python
Windows: C:/Users/<you>/miniconda3/envs/cs698b_nemo/python.exe
```

If `conda activate` says the environment is not found, run `conda env list` and use the folder shown for `cs698b_nemo`: add `/bin/python` (Linux) or `/python.exe` (Windows) to it. This happens when more than one conda (e.g. Miniconda and Miniforge) is installed.

**3b. Find `NEMO_MSDD_CHECKPOINT`**

```bash
# Linux
find ~/.cache/torch/NeMo -name diar_msdd_telephonic.nemo
```

```powershell
# Windows
(Get-ChildItem -Recurse "$env:USERPROFILE\.cache\torch\NeMo" -Filter diar_msdd_telephonic.nemo).FullName -replace '\\','/'
```

It prints one path, for example:

```
Linux:   /home/<you>/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/<hash>/diar_msdd_telephonic.nemo
Windows: C:/Users/<you>/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/<hash>/diar_msdd_telephonic.nemo
```

If it prints nothing, the model wasn't downloaded: run the download command from Step 2 again.

**3c. Put both paths in `app/.env`**

Open `app/.env` (Linux: `nano .env`; Windows: `notepad .env`) and add the two lines, pasting the output of 3a and 3b after the `=` (no quotes, no spaces):

```
NEMO_PYTHON=<output of 3a>
NEMO_MSDD_CHECKPOINT=<output of 3b>
```

Example (Linux):

```
NEMO_PYTHON=/home/alex/miniconda3/envs/cs698b_nemo/bin/python
NEMO_MSDD_CHECKPOINT=/home/alex/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo
```

> On Windows, use forward slashes `/` in `.env` (the commands above already print them that way). On Linux the app can auto-detect both paths, but setting them is safer. On Windows, `NEMO_PYTHON` **must** be set.

On a CPU-only machine also set:

```
DEVICE=cpu
COMPUTE_TYPE=int8
WHISPER_MODEL=small
```

**Check it works:** after transcribing a file, its result shows `diarization_model: diar_msdd_telephonic.nemo + TitaNet-Large`. If it shows `DIARIZATION_UNAVAILABLE`, the reason is in `diarization_blocker` (usually a wrong path in `.env`).

### Fallback: Hugging Face / Pyannote diarization

Use this **only if NeMo fails**. The app tries NeMo first and falls back to Pyannote when NeMo isn't available and `HF_TOKEN` is set. With neither, it still transcribes, but every word is labelled `Speaker`.

**1. Get a Hugging Face token**

1. Create a free account at <https://huggingface.co> (or log in).
2. Go to **Settings → Access Tokens** (<https://huggingface.co/settings/tokens>) and click **Create new token**.
3. Choose the **Read** token type, give it a name, and copy the token (it starts with `hf_`).

**2. Accept the terms of the 4 gated models**

The Pyannote models are **gated**: your token only works after you accept each model's user conditions. While logged in **with the same account that owns the token**, open each page below, fill in the short form and click **Agree and access repository**:

| # | Model | Used for |
|---|---|---|
| 1 | [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) | Speaker diarization pipeline (who spoke when) |
| 2 | [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0) | Speaker segmentation model used by the diarization pipeline |
| 3 | [`pyannote/voice-activity-detection`](https://hf.co/pyannote/voice-activity-detection) | Voice activity detection pipeline (`configs/config.yaml`) |
| 4 | [`pyannote/segmentation`](https://hf.co/pyannote/segmentation) | Segmentation model used by the VAD pipeline |

Skipping any of them causes a `403` / gated-model error when diarization runs.

**3. Add the token to `.env`**

Open `app/.env` in a text editor (Linux: `nano .env` or `gedit .env`; Windows: `notepad .env`), find this line:

```
# HF_TOKEN=
```

and replace it with your token (remove the leading `#`):

```
HF_TOKEN=hf_your_token_here
```

Save the file and restart the backend. The app loads `.env` automatically on startup.

> If `HF_TOKEN` is not set, the app also looks for `HUGGINGFACE_HUB_TOKEN` and then for a token saved by `huggingface-cli login`. The token is never logged or stored in the database.

### Other settings

All other settings have sensible defaults. Change them in the same `.env` file if needed:

| Variable | Default | Purpose |
|---|---|---|
| `NEMO_PYTHON` / `NEMO_MSDD_CHECKPOINT` | auto-detected (Linux) | NeMo diarization environment and checkpoint ([Step 3](#step-3-point-the-app-at-nemo)) |
| `HF_TOKEN` | *(none)* | Fallback: speaker labels via Pyannote when NeMo isn't available |
| `WHISPER_MODEL` | `large-v3` | Use `small` or `medium` for faster (less accurate) transcription on CPU |
| `DEVICE` | `auto` | `auto` uses the GPU if there is one; set `cpu` or `cuda` to force |
| `COMPUTE_TYPE` | `auto` | Use `int8` on CPU, `float16` on a GPU with ≥ 6 GB VRAM |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Backend address |
| `API_URL` | `http://127.0.0.1:8000` | Backend URL the dashboard connects to |
| `FRONTEND_PORT` | `8501` | Dashboard port |
| `STORAGE_DIR` | `storage` | Uploads, processed audio and results |
| `MAX_UPLOAD_SIZE_MB` | `250` | Upload size limit |
| `MAX_DURATION_SECONDS` | `1800` | Maximum media length (30 min) |
| `APP_DB_PATH` | `storage/app.db` | SQLite database location |

---

## 6. Running the app

The app needs **two terminals**: one for the backend, one for the dashboard. Activate the virtual environment in **both**.

### Linux

**Terminal 1: backend API**

```bash
cd Video-Transcription-with-Confidence-Analytics/app
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2: dashboard**

```bash
cd Video-Transcription-with-Confidence-Analytics/app
source .venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Windows (PowerShell)

**Terminal 1: backend API**

```powershell
cd Video-Transcription-with-Confidence-Analytics\app
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2: dashboard**

```powershell
cd Video-Transcription-with-Confidence-Analytics\app
.venv\Scripts\Activate.ps1
streamlit run frontend\streamlit_app.py --server.port 8501
```

> If Windows Firewall asks whether to allow Python to communicate on networks, you can click **Cancel**: the app only listens on `127.0.0.1` (your own computer).

### Open the app

Wait for `Application startup complete.` in Terminal 1, then open:

| What | URL |
|---|---|
| Dashboard | <http://localhost:8501> |
| API docs (Swagger UI, try requests live) | <http://127.0.0.1:8000/docs> |
| API docs (ReDoc, read-only reference) | <http://127.0.0.1:8000/redoc> |
| OpenAPI spec (JSON) | <http://127.0.0.1:8000/openapi.json> |

`app/redoc.html` is an offline snapshot of the same ReDoc page. Open it directly in a browser to read the API reference without starting the backend (it needs internet to load the ReDoc script). It does not update itself: regenerate it from `/openapi.json` after changing the API.

> **First run:** the first transcription downloads the Whisper `large-v3` model (~3 GB), which can take several minutes. Later runs reuse the cached model.
>
> **Speed on CPU:** expect roughly 3–4× the media length (a 1-minute video takes about 3–4 minutes). Set `WHISPER_MODEL=small` in `.env` for much faster, slightly less accurate results.

---

## 7. Using the dashboard

The page has two cards at the top: **New transcription** (upload) and **Recent recordings**.

1. **Upload a file:** choose a video or audio file and click **Upload and transcribe**. If you upload a file that was already transcribed, the saved result opens immediately instead of transcribing again.
2. **Wait for processing:** a progress bar shows the current stage. You can leave the page open while it works.
3. **Review the result** in the tabs:
   - **📝 Transcript:** each speaker turn is shown as a card. Low-confidence words are highlighted and marked ▲; medium ones are dotted-underlined. Hover a word to see its score. Use **✏️ Correct the transcript** to fix a turn's text or speaker, then click **Save correction**. Saving a turn unchanged marks it as verified-correct, which also helps calibration.
   - **📊 Analytics:** confidence over time, score distribution, per-turn confidence, speech timeline and calibration metrics.
   - **🧠 Learning:** what the system has learned: the vocabulary, the hint sent to Whisper on the next job, the starting vs learned calibration curve, and the corrections made to this recording. Click **Re-transcribe this recording with what was learned** to run the pipeline again with that vocabulary and calibration (this replaces your earlier manual edits to the recording).
   - **📋 Data:** sortable tables of turns and words, with CSV download.
   - **💾 Export:** download as JSON, plain text, SRT or WebVTT.
   - **🧩 Plugin results:** what each registered plugin (another team's service) sent back for this recording. See [Plugins](#plugins).
4. **Recent recordings:** pick any earlier recording from the list to reopen its result instantly. A recording that was uploaded but never processed shows a **Transcribe this recording** button.

Calibration starts re-fitting once about 20 words have been reviewed. The sidebar controls the colour scheme, text size and confidence thresholds.

### Plugins

A plugin is **another team's web service**, usually running on their laptop. It receives each finished transcript and answers with an **insight card**.

The switch at the top of the page has two sections: **🎬 Transcribe** and **🧩 Plugins**.

1. **Register:** open **🧩 Plugins**. You don't need to open a recording first. Enter the **team name**, the **plugin name** and the **plugin URL** (e.g. `http://<their-laptop-ip>:8300/insight`), then click **Register plugin**. Click **Send test** to check that the app can reach it.
2. **Transcribe** a video as usual. When the transcript is ready, the processing job sends it to every active plugin (`transcript.ready`) before it finishes ("Sending transcript to plugins" in the progress bar). A slow or broken plugin never makes the transcription fail.
3. **Open the recording's 🧩 Plugin results tab.** It shows one box per plugin with the team, plugin name, URL, time and status (● received / ▲ failed), the card it sent, and the **raw response received**. **↻ Send again** sends the transcript again.
4. **Correct a turn** in the Transcript tab, and the new version is sent to plugins in the background (`transcript.updated`).

Recordings transcribed before a plugin was registered are sent to it the first time you open their Plugin results tab.

Cards are validated and drawn by the dashboard itself. Plugins cannot send HTML, Markdown or colours. Flag levels reuse the confidence tiers (attention ▲, info ■, good ●), so every card follows the selected colourblind-safe scheme, and colour is never the only cue. Each chart also has a table version.

The full contract (an example event and an example card) is at `GET /api/v1/extensions/contract`. It is also shown in the Plugins section under "For plugin authors".

**Try it with the bundled example plugin** ("Talk balance"):

```bash
# from app/, in a third terminal
uvicorn extensions_examples.talk_balance:app --port 8300
```

Then register `http://127.0.0.1:8300/insight` in **🧩 Plugins** and click **Send test**. A fuller, standalone plugin that can run on another laptop is in `../video_transcription_plugin` (next to this project in Downloads).

---

## 8. Stopping the app

Press **`Ctrl+C`** in each terminal (press it again if the backend waits on a running job), then leave the virtual environment (same command on every OS):

```bash
deactivate
```

---

## 9. Running tests

```bash
# Linux
cd Video-Transcription-with-Confidence-Analytics/app
source .venv/bin/activate
pytest -q
```

```powershell
# Windows PowerShell
cd Video-Transcription-with-Confidence-Analytics\app
.venv\Scripts\Activate.ps1
pytest -q
```

The tests use a temporary database, so they never touch `storage/app.db` or jobs running on a live server. They mock the models, so no GPU or model download is needed.

### Benchmarks and test clips

Build short two-speaker test clips with exact ground truth from Mini LibriSpeech:

```bash
python scripts/make_librispeech_clips.py              # 5 clips, 60–120 s each
python scripts/make_librispeech_clips.py --n 8 --mp4  # 8 clips as .mp4 videos
```

Benchmark real inference on the benchmark recordings:

```bash
python scripts/run_benchmarks.py --model tiny --limit 1
```

---

## 10. REST API

Interactive docs: <http://127.0.0.1:8000/docs> (Swagger) and <http://127.0.0.1:8000/redoc> (ReDoc). In the paths below, `{id}` is a recording ID returned by the upload endpoint.

### Health

| Method | Endpoint | Description |
|:--|:--|:--|
| `GET` | `/api/v1/health` | Service status and engine configuration |

### Media

| Method | Endpoint | Description |
|:--|:--|:--|
| `POST` | `/api/v1/media/upload` | Upload and validate a media file (multipart field `file`) |
| `GET` | `/api/v1/media` | List ingested recordings |
| `GET` | `/api/v1/media/{id}/stream` | Stream audio for playback |

### Transcriptions

| Method | Endpoint | Description |
|:--|:--|:--|
| `POST` | `/api/v1/transcriptions` | Start a transcription job (returns `job_id`) |
| `GET` | `/api/v1/transcriptions/{job_or_rec_id}/status` | Job progress (0.0–1.0) and current stage; accepts a job ID or recording ID |
| `GET` | `/api/v1/transcriptions/{id}` | Full `TranscriptResult` JSON |
| `GET` | `/api/v1/transcriptions/{id}/export?format=` | Export as `txt`, `json`, `srt` or `vtt` |

`POST /api/v1/transcriptions` body (only `recording_id` is required):

| Field | Default | Meaning |
|---|---|---|
| `recording_id` | *(required)* | Recording to transcribe |
| `language` | `"en"` | Spoken language code |
| `enable_denoising` | `false` | Apply speech enhancement first |
| `initial_prompt` | `null` | Extra domain vocabulary hint for Whisper |
| `use_learned_vocabulary` | `true` | Prepend vocabulary learned from corrections to the Whisper prompt |
| `force` | `false` | Re-run even if a transcript already exists |

### Analytics

| Method | Endpoint | Description |
|:--|:--|:--|
| `GET` | `/api/v1/analytics/{id}` | Confidence tiers, ECE, Brier score, reliability diagram |

### Feedback and adaptive learning

| Method | Endpoint | Description |
|:--|:--|:--|
| `POST` | `/api/v1/feedback` | Submit a correction to one transcript turn (text and/or speaker). Updates the transcript, learns the corrected words as vocabulary, and re-fits the confidence calibrator. |
| `GET` | `/api/v1/feedback?recording_id=` | List corrections, newest first (`recording_id` is optional) |
| `DELETE` | `/api/v1/feedback/{feedback_id}` | Forget a correction; vocabulary and calibration are re-learned without it |
| `GET` | `/api/v1/feedback/vocabulary` | Domain terms learned from corrections, most frequently corrected first |
| `GET` | `/api/v1/adaptive/status` | What the system has learned: vocabulary, current Whisper prompt and calibration fit |

`POST /api/v1/feedback` body:

| Field | Required | Meaning |
|---|---|---|
| `recording_id` | yes | Recording being corrected |
| `original_text` | yes | Turn text as transcribed |
| `corrected_text` | yes | Fixed text (send it unchanged to mark the turn as correct) |
| `turn_id` | no | ID of the turn being corrected |
| `word_index` | no | Index of a single word, if correcting one word |
| `original_speaker` / `corrected_speaker` | no | Speaker fix |
| `notes` | no | Free-text note |

The response reports the `error_type` (`acoustic_lexical`, `diarization`, `compound` or `none`), `extracted_terms`, `words_changed` / `words_total`, `samples_added`, `vocabulary_size`, and the updated `calibration` state.

### Examples: Linux (bash)

```bash
# Upload, then start transcription
curl -F "file=@/path/to/video.mp4" http://127.0.0.1:8000/api/v1/media/upload
curl -X POST -H "Content-Type: application/json" \
     -d '{"recording_id":"<id>"}' http://127.0.0.1:8000/api/v1/transcriptions

# Progress, result, export
curl http://127.0.0.1:8000/api/v1/transcriptions/<id>/status
curl "http://127.0.0.1:8000/api/v1/transcriptions/<id>/export?format=srt" -o transcript.srt

# Submit a correction, then see what was learned
curl -X POST -H "Content-Type: application/json" \
     -d '{"recording_id":"<id>","turn_id":"<turn_id>","original_text":"the cuber netes cluster","corrected_text":"the Kubernetes cluster"}' \
     http://127.0.0.1:8000/api/v1/feedback
curl http://127.0.0.1:8000/api/v1/feedback/vocabulary
curl http://127.0.0.1:8000/api/v1/adaptive/status

# Re-transcribe using the learned vocabulary
curl -X POST -H "Content-Type: application/json" \
     -d '{"recording_id":"<id>","force":true}' http://127.0.0.1:8000/api/v1/transcriptions
```

### Examples: Windows (PowerShell)

In Windows PowerShell 5, `curl` is an alias for `Invoke-WebRequest`, so use `curl.exe` for uploads and `Invoke-RestMethod` for JSON:

```powershell
$api = "http://127.0.0.1:8000/api/v1"

# Upload, then start transcription
curl.exe -F "file=@C:\path\to\video.mp4" "$api/media/upload"
Invoke-RestMethod -Method Post -Uri "$api/transcriptions" -ContentType "application/json" `
  -Body (@{ recording_id = "<id>" } | ConvertTo-Json)

# Progress, result, export
Invoke-RestMethod "$api/transcriptions/<id>/status"
Invoke-WebRequest "$api/transcriptions/<id>/export?format=srt" -OutFile transcript.srt

# Submit a correction, then see what was learned
Invoke-RestMethod -Method Post -Uri "$api/feedback" -ContentType "application/json" -Body (@{
  recording_id   = "<id>"
  turn_id        = "<turn_id>"
  original_text  = "the cuber netes cluster"
  corrected_text = "the Kubernetes cluster"
} | ConvertTo-Json)
Invoke-RestMethod "$api/feedback/vocabulary"
Invoke-RestMethod "$api/adaptive/status"

# Delete a correction
Invoke-RestMethod -Method Delete -Uri "$api/feedback/<feedback_id>"
```

---

## 11. Architecture

```text
Streamlit dashboard (:8501)
        │  HTTP / REST v1
        ▼
FastAPI backend (:8000)
        ├── Services: upload, hashing, background job queue (one job at a time), restart recovery
        ├── SQLite persistence (storage/app.db)
        ├── Speech engine (engine/)
        │     ├── Preprocessing: FFmpeg → 16 kHz mono WAV, loudness normalisation
        │     ├── Voice activity detection
        │     ├── ASR: Faster-Whisper large-v3 (condition_on_previous_text=False)
        │     │        prompt = learned vocabulary + optional initial_prompt
        │     ├── Word alignment: Wav2Vec2
        │     ├── Diarization: NeMo MSDD + TitaNet (isolated env) → Pyannote 3.1 fallback
        │     └── Word-to-speaker fusion (midpoint / max overlap with collar)
        └── Adaptive subsystem (adaptive/)
              ├── Confidence calibration (isotonic regression, Brier, ECE)
              │        re-fitted from words reviewed through /feedback
              └── Feedback loop: error classification, domain vocabulary extraction
```

---

## 12. Project layout

```text
app/
├── backend/           FastAPI app, API endpoints, services, SQLite models
├── engine/            Speech pipeline (Whisper, alignment, diarization)
├── adaptive/          Confidence calibration and feedback loop
├── frontend/          Streamlit dashboard and UI components (incl. Learning tab)
├── configs/           Pipeline parameters (config.yaml)
├── scripts/           DB init, benchmark CLI, LibriSpeech test-clip builder
├── storage/           Uploads, processed audio, results, app.db (created at runtime)
├── tests/             pytest suite
├── .streamlit/        Dashboard theme
├── .env.example       Configuration template
├── redoc.html         Offline snapshot of the API reference
├── pyproject.toml     Package metadata and dependencies
└── requirements.txt   Installs the package (-e .[dev])
```

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar shows **Backend offline** | Start the backend (Terminal 1) and check the URL in the sidebar is `http://127.0.0.1:8000`. |
| Every word is labelled `Speaker` | Diarization isn't configured. Set up NeMo ([Step 2](#step-2-set-up-nemo-diarization-recommended)), or use the [Hugging Face fallback](#fallback-hugging-face--pyannote-diarization). The reason is in the result's `diarization_blocker`. |
| `NeMo Python executable not found` / `checkpoint not found` | Fix `NEMO_PYTHON` / `NEMO_MSDD_CHECKPOINT` in `app/.env` ([Step 3](#step-3-point-the-app-at-nemo)) and restart the backend. |
| NeMo diarization times out on long files | On CPU the NeMo step can exceed its 300 s limit. Raise `timeout_seconds` in `engine/nemo_diarizer.py`, or use a GPU. |
| `403` / gated model error (fallback) | You haven't accepted the terms on **all 4** Pyannote model pages with the account that owns the token. |
| `401` / invalid token error | The token in `app/.env` is wrong, expired or revoked. Create another **Read** token and update `HF_TOKEN`. |
| `FFmpeg preprocessing failed` | Install FFmpeg ([section 3](#3-install-system-packages)) and make sure `ffmpeg -version` works in the same terminal. |
| Linux: `ensurepip is not available` when creating the venv | `sudo apt install python3-venv` (or `python3.X-venv` for your exact version). |
| Windows: `running scripts is disabled on this system` | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once. |
| Windows: `python`/`ffmpeg` is not recognized | Reopen PowerShell after installing. If it still fails, add Python / `C:\ffmpeg\bin` to `PATH`. Before the venv is active, use `py` instead of `python`. |
| Windows: GPU not used | The default Windows PyTorch wheel is CPU-only. Reinstall with the CUDA index URL ([section 4](#windows-powershell)). |
| Job stuck at *Queued* | Only one job runs at a time; it waits for the current job to finish. |
| Job failed: *interrupted by application restart* | The backend was restarted mid-job. Click **Retry transcription**. |
| Learning tab: calibration not fitted | It needs about 20 reviewed words. Correct or verify more turns. |
| Re-transcribe button is disabled | Nothing has been learned yet. Save at least one correction that changes a word. |
| Transcription very slow | Expected on CPU with `large-v3`. Set `WHISPER_MODEL=small` in `.env`, or use a GPU (`DEVICE=cuda`). |
| `torchcodec is not installed correctly` warning | Harmless: the app loads audio itself and doesn't use torchcodec. |
| Port already in use | Another instance is running. Stop it, or use `--port 8001` / `--server.port 8502` (and update the backend URL in the sidebar). |

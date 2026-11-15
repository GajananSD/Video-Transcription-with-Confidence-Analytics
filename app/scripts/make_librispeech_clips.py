#!/usr/bin/env python3
"""Build short two-speaker test clips with exact ground truth from Mini LibriSpeech.

Mini LibriSpeech (OpenSLR resource 31, ``dev-clean-2``) contains short read-speech
utterances with human transcripts. This script joins utterances from two different
speakers, taking turns, into 1-2 minute "conversations". Because the script places
every utterance itself, it knows exactly what was said, by whom, and when.

For each clip it writes::

    raw_audio/<name>.wav                 16 kHz mono audio (or .mp4 with --mp4)
    ground_truth/<name>.txt              plain transcript (used by run_benchmarks.py for WER)
    ground_truth_json/<name>.json        {"file", "segments": [{speaker, start, end, text}]}
    ground_truth_rttm/<name>.rttm        speaker timing reference for diarization scoring

Usage (from app/, with the venv active)::

    python scripts/make_librispeech_clips.py                 # 5 clips, 60-120 s each
    python scripts/make_librispeech_clips.py --n 8 --mp4     # 8 clips as .mp4 videos
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

URL = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
SR = 16000
REPO_ROOT = Path(__file__).resolve().parents[2]


def download_dataset(cache_dir: Path) -> Path:
    """Download and extract Mini LibriSpeech once; return the dev-clean-2 directory."""
    data_dir = cache_dir / "LibriSpeech" / "dev-clean-2"
    if data_dir.exists():
        return data_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "dev-clean-2.tar.gz"
    if not archive.exists():
        print(f"Downloading {URL} (~126 MB) ...")
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(archive)
    print("Extracting ...")
    with tarfile.open(archive) as tar:
        tar.extractall(cache_dir, filter="data")
    if not data_dir.exists():
        raise SystemExit(f"Unexpected archive layout: {data_dir} not found")
    return data_dir


def load_utterances(data_dir: Path) -> dict[str, list[dict]]:
    """Map speaker id -> list of {path, text} from the LibriSpeech *.trans.txt files."""
    by_speaker: dict[str, list[dict]] = {}
    for trans in sorted(data_dir.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            flac = trans.parent / f"{utt_id}.flac"
            if flac.exists() and text.strip():
                by_speaker.setdefault(utt_id.split("-")[0], []).append({"path": flac, "text": text.strip()})
    return by_speaker


def load_genders(data_dir: Path) -> dict[str, str]:
    """Speaker id -> 'M'/'F' if SPEAKERS.TXT is present (helps pick distinct-sounding pairs)."""
    genders: dict[str, str] = {}
    for cand in (data_dir.parent / "SPEAKERS.TXT", data_dir / "SPEAKERS.TXT"):
        if cand.exists():
            for line in cand.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    genders[parts[0]] = parts[1]
    return genders


def build_clip(rng: random.Random, utts_a: list[dict], utts_b: list[dict], min_s: float, max_s: float):
    """Alternate utterances from speaker A and B with short pauses until min_s..max_s is reached."""
    pool = {"A": utts_a[:], "B": utts_b[:]}
    for p in pool.values():
        rng.shuffle(p)
    audio: list[np.ndarray] = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    t = 0.5
    segments = []
    turn = "A"
    while t < min_s and pool[turn]:
        utt = pool[turn].pop()
        wav, sr = sf.read(str(utt["path"]), dtype="float32")
        if sr != SR:
            raise SystemExit(f"Unexpected sample rate {sr} in {utt['path']}")
        dur = len(wav) / SR
        if t + dur > max_s:
            break
        segments.append({"speaker": f"Speaker_{0 if turn == 'A' else 1}", "start": round(t, 3),
                         "end": round(t + dur, 3), "text": utt["text"]})
        audio.append(wav)
        t += dur
        gap = rng.uniform(0.3, 0.9)
        audio.append(np.zeros(int(gap * SR), dtype=np.float32))
        t += gap
        turn = "B" if turn == "A" else "A"
    return np.concatenate(audio), segments


def write_outputs(name: str, audio: np.ndarray, segments: list[dict], mp4: bool) -> Path:
    raw_dir = REPO_ROOT / "raw_audio"
    gt_dir = REPO_ROOT / "ground_truth"
    gt_json_dir = REPO_ROOT / "ground_truth_json"
    rttm_dir = REPO_ROOT / "ground_truth_rttm"
    for d in (raw_dir, gt_dir, gt_json_dir, rttm_dir):
        d.mkdir(parents=True, exist_ok=True)

    wav_path = raw_dir / f"{name}.wav"
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(str(wav_path), (audio / peak * 0.9).astype(np.float32), SR, subtype="PCM_16")
    media_path = wav_path
    if mp4:
        media_path = raw_dir / f"{name}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=1",
             "-i", str(wav_path), "-shortest", "-c:v", "libx264", "-tune", "stillimage",
             "-c:a", "aac", "-b:a", "128k", str(media_path)],
            check=True,
        )
        wav_path.unlink()

    (gt_dir / f"{name}.txt").write_text(" ".join(s["text"] for s in segments) + "\n", encoding="utf-8")
    (gt_json_dir / f"{name}.json").write_text(
        json.dumps({"file": name, "source": "Mini LibriSpeech dev-clean-2", "segments": segments}, indent=2),
        encoding="utf-8",
    )
    (rttm_dir / f"{name}.rttm").write_text(
        "".join(
            f"SPEAKER {name} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n"
            for s in segments
        ),
        encoding="utf-8",
    )
    return media_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5, help="Number of clips to create")
    parser.add_argument("--min-seconds", type=float, default=60.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--mp4", action="store_true", help="Write .mp4 video (black frame) instead of .wav")
    parser.add_argument("--seed", type=int, default=698)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "mini_librispeech")
    args = parser.parse_args()

    if args.mp4 and not shutil.which("ffmpeg"):
        raise SystemExit("--mp4 needs ffmpeg on PATH")

    rng = random.Random(args.seed)
    data_dir = download_dataset(args.cache_dir)
    by_speaker = load_utterances(data_dir)
    genders = load_genders(data_dir)
    speakers = sorted(s for s, u in by_speaker.items() if len(u) >= 8)
    rng.shuffle(speakers)
    print(f"Found {sum(len(u) for u in by_speaker.values())} utterances from {len(by_speaker)} speakers.")

    used: set[str] = set()
    made = 0
    for spk_a in speakers:
        if made >= args.n:
            break
        if spk_a in used:
            continue
        # Prefer a partner of the other gender so the two voices are easy to tell apart.
        partners = [s for s in speakers if s not in used and s != spk_a]
        if genders.get(spk_a):
            partners.sort(key=lambda s: genders.get(s) == genders.get(spk_a))
        if not partners:
            break
        spk_b = partners[0]
        audio, segments = build_clip(rng, by_speaker[spk_a], by_speaker[spk_b], args.min_seconds, args.max_seconds)
        duration = len(audio) / SR
        if duration < args.min_seconds * 0.8 or len(segments) < 4:
            continue  # not enough material for these two speakers
        used.update({spk_a, spk_b})
        made += 1
        name = f"librispeech_conversation_{made:02d}"
        path = write_outputs(name, audio, segments, args.mp4)
        n_words = sum(len(s["text"].split()) for s in segments)
        print(f"  {path.relative_to(REPO_ROOT)}  {duration:5.1f}s  {len(segments)} turns  {n_words} words  "
              f"(speakers {spk_a} + {spk_b})")

    print(f"\nDone: {made} clips. Ground truth in ground_truth/, ground_truth_json/, ground_truth_rttm/.")
    print("Benchmark them with:  python scripts/run_benchmarks.py --model tiny")


if __name__ == "__main__":
    main()

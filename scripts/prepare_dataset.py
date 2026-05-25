#!/usr/bin/env python3
"""
Kokoro TTS: Dataset Preparation Pipeline
========================================
Standardizes a speech dataset for Kokoro training. 

Functionality:
1. Audio: Converts files to 24kHz, mono, s16 PCM in-place.
2. Text: Generates IPA phonemes using the custom Phonemizer.
3. Output: Generates metadata.csv and phonemes.csv for training.

Usage:
    uv run python scripts/prepare_dataset.py \
        --transcriptions path/to/text.txt \
        --audio-dir path/to/wavs \
        --lang pl
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from loguru import logger

# Add project root to sys.path to allow importing tools.phonemes
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

try:
    from tools.phonemes.phonemizer import Phonemizer
except ImportError:
    logger.error("Failed to import Phonemizer from tools.phonemes.phonemizer")
    raise


# ─── Audio Processing ─────────────────────────────────────────────────────────

def standardize_audio(wav_path: Path) -> bool:
    """
    Converts audio to 24kHz, mono, 16-bit PCM in-place using FFmpeg.
    Returns True if successful, False otherwise.
    """
    temp_wav = wav_path.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(wav_path),
                "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                str(temp_wav)
            ],
            check=True, capture_output=True
        )
        os.replace(temp_wav, wav_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed for {wav_path.name}: {e.stderr.decode().strip()}")
        if temp_wav.exists():
            temp_wav.unlink()
        return False


# ─── Text Processing ──────────────────────────────────────────────────────────

def load_transcriptions(path: Path) -> List[str]:
    """Loads transcriptions from a file, one per line."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_pipeline(trans_path: Path, audio_dir: Path, lang: str, out_dir: Path):
    """Orchestrates the conversion and phonemization loop."""
    
    # 1. Initialize Tools
    logger.info(f"Initializing Phonemizer for language: {lang}")
    try:
        phonemizer = Phonemizer(lang_code=lang)
    except Exception as e:
        logger.critical(f"Failed to initialize Phonemizer: {e}")
        sys.exit(1)

    # 2. Load Data and Audio Files
    texts = load_transcriptions(trans_path)
    
    # Natural sort for audio files (ex. clone_1.wav, clone_2.wav, ..., clone_10.wav)
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', str(s))]
    
    wav_paths = sorted(audio_dir.glob("*.wav"), key=natural_sort_key)
    
    if len(wav_paths) != len(texts):
        logger.error(f"Mismatch: Found {len(wav_paths)} audio files but {len(texts)} transcriptions.")
        if len(wav_paths) > len(texts):
            logger.warning("Truncating audio list to match transcriptions.")
            wav_paths = wav_paths[:len(texts)]
        else:
            logger.warning("Truncating transcription list to match audio files.")
            texts = texts[:len(wav_paths)]

    logger.info(f"Loaded {len(texts)} pairs from {trans_path.name} and {audio_dir}")

    meta_rows = []
    phoneme_rows = []
    error_count = 0

    # 3. Processing Loop
    for wav_path, text in tqdm(zip(wav_paths, texts), total=len(texts), desc="Processing Dataset"):
        wav_name = wav_path.name
        audio_id = wav_path.stem

        # Step A: Audio Conversion (In-place)
        if not standardize_audio(wav_path):
            error_count += 1
            continue

        # Step B: Phonemization
        try:
            phonemes = phonemizer.phonemize(text)
        except Exception as e:
            logger.error(f"Phonemization failed for {audio_id}: {e}")
            error_count += 1
            continue

        # Step C: Accumulate Results
        # Using format compatible with StyleTTS2 trainer (filename|text|speaker_id)
        meta_rows.append(f"{wav_name}|{text}|0")
        phoneme_rows.append(f"{wav_name}|{phonemes}")

    # 4. Save Outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = out_dir / "metadata.csv"
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write("filename|text|speaker\n")
        f.write("\n".join(meta_rows) + "\n")

    phoneme_path = out_dir / "phonemes.csv"
    with open(phoneme_path, 'w', encoding='utf-8') as f:
        f.write("filename|ipa\n")
        f.write("\n".join(phoneme_rows) + "\n")

    logger.success(f"Processing complete!")
    logger.info(f"  Total processed: {len(meta_rows)}")
    logger.info(f"  Errors/Missing:  {error_count}")
    logger.info(f"  Output directory: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro TTS: Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required Arguments
    parser.add_argument("--transcriptions", required=True, type=Path, help="Path to transcriptions.txt (ID <tab> TEXT)")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory containing audio files to process in-place")
    parser.add_argument("--lang", required=True, help="Language code for phonemization (e.g., 'pl', 'de')")
    
    # Optional Arguments
    parser.add_argument("--output-dir", default=Path("./dataset"), type=Path, help="Directory to save metadata.csv and phonemes.csv")

    args = parser.parse_args()

    # Validation
    if not args.transcriptions.exists():
        logger.error(f"Transcriptions file not found: {args.transcriptions}")
        sys.exit(1)
    if not args.audio_dir.is_dir():
        logger.error(f"Audio directory not found: {args.audio_dir}")
        sys.exit(1)

    try:
        run_pipeline(args.transcriptions, args.audio_dir, args.lang, args.output_dir)
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()

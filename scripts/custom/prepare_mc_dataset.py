#!/usr/bin/env python3
"""
Prepare MC Speech Dataset for Kokoro TTS Training
================================================

This script processes the MC Speech Dataset (Polish) for training.
It reads transcriptions, phonemizes text using espeak-ng, applies 
language-specific phoneme fixups, and converts audio files to the 
required format (24kHz, mono, 16-bit).

Usage:
    uv run python scripts/custom/prepare_mc_dataset.py \
        --input /path/to/transcriptions.txt \
        --audio-dir /path/to/wavs \
        --output /path/to/train_dataset.csv
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from tools.phonemes.phonemizer import Phonemizer

def main():
    parser = argparse.ArgumentParser(description="Process MC Speech Dataset for Kokoro")
    parser.add_argument("--input", required=True, help="Path to transcriptions.txt")
    parser.add_argument("--audio-dir", help="Optional: Directory containing .wav files for format conversion")
    parser.add_argument("--meta-output", help="Optional: Path to metadata.csv (wav_name|text|0)")
    parser.add_argument("--phoneme-output", help="Optional: Path to phonemes.csv (wav_name|phonemes|0)")
    args = parser.parse_args()

    input_path = Path(args.input)
    audio_dir = Path(args.audio_dir) if args.audio_dir else None

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)
    if audio_dir and not audio_dir.exists():
        print(f"Error: Audio directory {audio_dir} not found.")
        sys.exit(1)
    if not args.meta_output and not args.phoneme_output:
        print("Warning: Neither --meta-output nor --phoneme-output specified. No output files will be created.")

    # Initialize Phonemizer
    print("Initializing Phonemizer 'pl'...")
    phonemizer = Phonemizer(lang_code="pl")

    # Load entries
    entries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))

    print(f"Processing {len(entries)} entries...")

    phoneme_rows = []
    meta_rows = []
    
    for audio_id, text in tqdm(entries, desc="Preparing"):
        wav_name = f"{audio_id}.wav"
        
        # 1. Convert Audio (Optional - only if audio_dir is provided)
        if audio_dir:
            wav_path = audio_dir / wav_name
            if wav_path.exists():
                # Format: 24kHz, 1 channel (mono), 16-bit PCM
                temp_wav = wav_path.with_suffix(".tmp.wav")
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(wav_path),
                            "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                            str(temp_wav)
                        ],
                        capture_output=True, check=True
                    )
                    os.replace(temp_wav, wav_path)
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {audio_id}: {e.stderr.decode()}")
                    if temp_wav.exists():
                        temp_wav.unlink()

        # 2. Phonemize and Fixup
        try:
            phonemes = phonemizer.phonemize(text)
        except Exception as e:
            print(f"Error phonemizing {audio_id}: {e}")
            continue

        # 3. Store row data
        if args.phoneme_output:
            phoneme_rows.append(f"{wav_name}|{phonemes}")
        if args.meta_output:
            meta_rows.append(f"{wav_name}|{text}|0")

    # Write output CSVs
    if args.meta_output:
        print(f"Writing metadata to {args.meta_output}...")
        with open(args.meta_output, 'w', encoding='utf-8') as f:
            for row in meta_rows:
                f.write(row + "\n")

    if args.phoneme_output:
        print(f"Writing phonemes to {args.phoneme_output}...")
        with open(args.phoneme_output, 'w', encoding='utf-8') as f:
            for row in phoneme_rows:
                f.write(row + "\n")

    print("Done!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert VoxCeleb2 .m4a files (under dev/aac/) to .wav format using ffmpeg.
Output is saved under dev/wav/, preserving the same directory structure.

Example:
    voxceleb2/
    ├── dev/
    │   ├── aac/
    │   │   └── id00012/abc123/00001.m4a
    │   └── wav/
    │       └── id00012/abc123/00001.wav
"""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# ================================================================
# USER SETUP
VOXCELEB2_ROOT = Path("/path/to/vox2")  # <-- change this
NUM_WORKERS = os.cpu_count() or 8            # number of parallel conversions
# ================================================================

def get_m4a_files(aac_dir: Path):
    """Recursively find all .m4a files under the AAC directory."""
    return list(aac_dir.rglob("*.m4a"))

def m4a_to_wav(m4a_path: Path, aac_dir: Path, wav_dir: Path):
    """Convert a single .m4a file to .wav using ffmpeg."""
    rel_path = m4a_path.relative_to(aac_dir)
    wav_path = wav_dir / rel_path.with_suffix(".wav")
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists():
        return f"[EXISTS] {wav_path}"

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(m4a_path),
        "-ar", "16000",  # resample to 16 kHz
        "-ac", "1",      # mono
        str(wav_path)
    ]

    try:
        subprocess.run(cmd, check=True)
        return f"[OK] {wav_path}"
    except subprocess.CalledProcessError:
        return f"[FAIL] {m4a_path}"

def main():
    aac_dir = VOXCELEB2_ROOT / "dev" / "aac"
    wav_dir = VOXCELEB2_ROOT / "dev" / "wav"

    if not aac_dir.exists():
        raise FileNotFoundError(f"AAC directory not found: {aac_dir}")

    m4a_files = get_m4a_files(aac_dir)
    print(f"🎧 Found {len(m4a_files)} .m4a files under {aac_dir}")

    results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(m4a_to_wav, f, aac_dir, wav_dir): f for f in m4a_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            result = future.result()
            if result and not result.startswith("[OK]"):
                print(result)
            results.append(result)

    ok_count = sum(1 for r in results if r and r.startswith("[OK]"))
    fail_count = sum(1 for r in results if r and r.startswith("[FAIL]"))
    print(f"\nDone! {ok_count} converted, {fail_count} failed.")
    print(f"WAV files saved under: {wav_dir.resolve()}")

if __name__ == "__main__":
    main()

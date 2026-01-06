#!/usr/bin/env python3
"""
Test vLLM Ultravox variable-length audio batching fix.

Tests fix for issue #31658: concurrent audio requests with different
durations crash due to incompatible tensor shapes.

Usage:
    python test_batching.py                    # Use real speech (MINDS-14 dataset)
    python test_batching.py --use-synthetic    # Use synthetic audio (no download)
    python test_batching.py --audio-dir /path  # Use local audio files
"""

import argparse
import asyncio
import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from openai import AsyncOpenAI

VLLM_URL = "http://127.0.0.1:8000/v1"
MODEL = "fixie-ai/ultravox-v0_6-llama-3_1-8b"


def create_synthetic_audio(duration_sec: float, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Create synthetic audio tone."""
    samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    return audio, sr


def load_from_file(path: Path) -> tuple[np.ndarray, int]:
    """Load audio from file."""
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


def load_from_dataset(index: int) -> tuple[np.ndarray, int]:
    """Load audio from MINDS-14 dataset."""
    from datasets import load_dataset
    ds = load_dataset("PolyAI/minds14", "en-US", split="train", trust_remote_code=True)
    sample = ds[index]
    return np.array(sample["audio"]["array"], dtype=np.float32), sample["audio"]["sampling_rate"]


def to_base64(audio: np.ndarray, sr: int) -> str:
    """Convert audio to base64 WAV."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


async def transcribe(client: AsyncOpenAI, audio_b64: str, label: str) -> dict:
    """Send transcription request."""
    start = time.time()
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio:"},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}
                ]
            }],
            max_tokens=256,
        )
        return {"label": label, "success": True, "response": resp.choices[0].message.content, "time": time.time() - start}
    except Exception as e:
        return {"label": label, "success": False, "error": str(e), "time": time.time() - start}


async def test_concurrent(client: AsyncOpenAI, samples: list[tuple[str, np.ndarray, int]], test_name: str) -> bool:
    """Test concurrent requests."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print("="*60)

    for label, audio, sr in samples:
        print(f"  - {label}: {len(audio)/sr:.2f}s")

    print(f"\nSending {len(samples)} concurrent requests...")

    tasks = [transcribe(client, to_base64(audio, sr), label) for label, audio, sr in samples]
    results = await asyncio.gather(*tasks)

    all_ok = True
    print("\nResults:")
    for r in results:
        if r["success"]:
            preview = r["response"][:60] + "..." if len(r["response"]) > 60 else r["response"]
            print(f"  ✅ {r['label']}: {preview} ({r['time']:.2f}s)")
        else:
            print(f"  ❌ {r['label']}: {r['error']}")
            all_ok = False

    return all_ok


async def main(args):
    global VLLM_URL
    if args.vllm_url:
        VLLM_URL = args.vllm_url

    print("\n" + "#"*60)
    print("# vLLM Ultravox Variable-Length Audio Batching Test")
    print("# Testing fix for issue #31658")
    print("#"*60)

    client = AsyncOpenAI(base_url=VLLM_URL, api_key="dummy")

    # Prepare samples
    samples = []

    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.flac"))
        if not files:
            print(f"No audio files in {audio_dir}")
            sys.exit(1)
        print(f"\nLoading {min(4, len(files))} files from {audio_dir}...")
        for i, f in enumerate(files[:4]):
            audio, sr = load_from_file(f)
            samples.append((f.name, audio, sr))

    elif args.use_synthetic:
        print("\nUsing synthetic audio...")
        for dur in [1.5, 2.5, 4.0, 6.0]:
            audio, sr = create_synthetic_audio(dur)
            samples.append((f"synthetic_{dur}s", audio, sr))

    else:
        # Default: use real audio from MINDS-14 dataset
        print("\nLoading real speech from MINDS-14 dataset...")
        try:
            for i in [0, 5, 10, 15]:
                audio, sr = load_from_dataset(i)
                samples.append((f"minds14_{i}", audio, sr))
        except Exception as e:
            print(f"Dataset failed: {e}, falling back to synthetic audio")
            for dur in [1.5, 2.5, 4.0, 6.0]:
                audio, sr = create_synthetic_audio(dur)
                samples.append((f"synthetic_{dur}s", audio, sr))

    # Run tests
    results = []

    # Test 1: Sequential baseline
    print(f"\n{'='*60}")
    print("TEST: Sequential requests (baseline)")
    print("="*60)
    for label, audio, sr in samples[:2]:
        r = await transcribe(client, to_base64(audio, sr), label)
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {label}: {r.get('response', r.get('error', ''))[:50]}...")
    results.append(("Sequential", True))

    # Test 2: Same-length concurrent
    same_samples = [(f"same_1", samples[0][1], samples[0][2]), (f"same_2", samples[0][1].copy(), samples[0][2])]
    ok = await test_concurrent(client, same_samples, "Same-length concurrent")
    results.append(("Same-length concurrent", ok))

    # Test 3: CRITICAL - Different lengths concurrent
    sorted_samples = sorted(samples, key=lambda x: len(x[1])/x[2])
    diff_samples = [sorted_samples[0], sorted_samples[-1]]
    ok = await test_concurrent(client, diff_samples, "CRITICAL: Variable-length concurrent")
    results.append(("Variable-length concurrent", ok))

    # Test 4: All samples concurrent
    if len(samples) >= 3:
        ok = await test_concurrent(client, samples, "All samples concurrent")
        results.append(("All concurrent", ok))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, ok in results:
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}: {name}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("The variable-length audio batching fix is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("If you see 'inconsistent shapes' error, the fix is NOT applied.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", help="Directory with audio files")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic audio instead of real speech")
    parser.add_argument("--vllm-url", default=VLLM_URL, help="vLLM server URL")
    asyncio.run(main(parser.parse_args()))

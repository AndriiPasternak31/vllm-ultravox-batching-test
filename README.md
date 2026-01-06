# vLLM Ultravox Variable-Length Audio Batching Test

This project tests the fix for [vLLM issue #31658](https://github.com/vllm-project/vllm/issues/31658) where concurrent audio requests with different durations crash the server.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install vLLM with the fix
pip install git+https://github.com/AndriiPasternak31/vllm.git@fix/ultravox-variable-length-audio-batching

# 3. Start vLLM server (in one terminal)
./start_server.sh

# 4. Run tests (in another terminal)
python test_batching.py
```

## What This Tests

The bug occurs when sending concurrent audio transcription requests with **different durations**:

```
ValueError: data contains inconsistent shapes: torch.Size([128, 325]) (index 0) vs torch.Size([128, 666]) (index 1)
```

The fix introduces `MultiModalListField` that properly handles variable-length audio tensors.

## Test Options

```bash
# Use synthetic audio (default, no download needed)
python test_batching.py

# Use real speech from MINDS-14 dataset
python test_batching.py --use-dataset

# Use your own audio files
python test_batching.py --audio-dir /path/to/wav/files

# Custom vLLM server URL
python test_batching.py --vllm-url http://localhost:8000/v1
```

## Expected Output

With the fix applied:
```
🎉 ALL TESTS PASSED!
The variable-length audio batching fix is working correctly.
```

Without the fix:
```
❌ SOME TESTS FAILED!
ValueError: data contains inconsistent shapes...
```

## Files

- `test_batching.py` - Main test script
- `start_server.sh` - Script to start vLLM server
- `requirements.txt` - Python dependencies

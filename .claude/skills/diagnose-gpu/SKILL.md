---
name: diagnose-gpu
description: Diagnose GPU use in this repository: CUDA 12 and cuDNN 9 discovery, a run that stops while building the GPU model, and the per-file CPU retry. Use when transcription did not use the NVIDIA GPU, is unexpectedly slow, exits with a CUDA, cuBLAS, cuDNN, ctranslate2 or driver error, switched to CPU after processing began, or before changing the CUDA setup.
---

# Diagnose GPU use

1. Confirm whether `--cpu` was passed. It deliberately bypasses GPU model construction and reports that choice to stderr.
2. Run `uv run transcribe --debug`. This checks native-library discovery without loading a model. GPU startup requires both CUDA 12 and cuDNN 9 library names; CUDA 11 or 13 and cuDNN 8 do not satisfy it.
3. If either library is missing, use the supported wheel setup:

   ```console
   uv sync --extra cuda
   uv run transcribe --debug
   ```

   On Windows, the process prepends wheel `bin` directories to `PATH`; wheel libraries take precedence over `CUDA_PATH` and `CUDNN_PATH`. Without the wheels, those variables or the inherited `PATH` must identify directories containing the required DLLs. On Linux, the process preloads wheel libraries from their `lib` directories. It does not search guessed system CUDA locations.
4. After both libraries are found, run `uv run transcribe --test`. This is the first real check of model construction, decoding, and device use. Do not start a full batch for diagnosis.
5. Classify the failure by when it occurs:
   - GPU model construction fails before processing: the command exits 1 and does not fall back automatically. Preserve the underlying error. Check the NVIDIA GPU and driver as well as CUDA discovery. Use `--cpu` only as a deliberate alternative.
   - A loaded GPU model fails on a file: stderr reports `GPU transcription failed` and retries that file on CPU. A successful retry disables the GPU for the rest of the run. A failed retry marks only that file failed; the next file still tries the GPU.

Do not infer GPU use from speed alone. A CPU retry may take much longer. Use the startup output and stderr state transitions as evidence.

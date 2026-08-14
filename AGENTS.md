# recording-transcriber

This repository contains a command-line tool that batch-transcribes audio and
video with faster-whisper. It writes plain transcripts under `transcripts/` and
timestamped transcripts under `transcripts_timestamps/`, mirroring the input
tree under `recordings/`.

This repository is public. Never write secrets, personal data, machine or
network details, or absolute paths outside the repository into tracked files.
Do not commit recordings, transcripts, model weights, `.env`, or files under
`local/`.

## Project map

- `src/main.py`: CLI, CUDA library setup, and batch orchestration.
- `src/transcribe.py`: input scan, output naming, and model wrapper.
- `tests/`: fake-based tests for naming, paths, GPU fallback, and model state.
  Tests must not download or load a real model.
- `README.md`: user-facing installation and options.
- `.agents/skills/<name>/SKILL.md`: frontmatter plus a pointer to
  `.claude/skills/<name>/SKILL.md`, which holds the single copy of the skill
  body. Read the pointed-to file. Edit that copy, not the pointer.

A skill description lists example phrasings, not the only way in. Load the
skill whose subject matches the work, however the request was worded.

`recordings/`, `transcripts/`, `transcripts_timestamps/`, and `notes/` are
gitignored and created by the first `uv run transcribe`.

## Environment and checks

Use uv for dependencies and for running commands. Do not use pip directly.

```console
uv sync
uv run transcribe --debug
uv run transcribe --test
```

For development checks, use the configured tools without adding formatters or
changing their configuration:

```console
uv run pyright
uv run pytest -q -p no:cacheprovider --basetemp=<temporary-directory-outside-repository>/pytest-tmp
```

Transcription can take minutes per file. A full batch is a heavy run. Do not
start `uv run transcribe` on a populated recordings tree unless the user asks.
Do not run it in the background. `--debug` does not load a model, and `--test`
processes only the first pending input, so use them for setup checks.

## CUDA on Windows and Linux

GPU use requires an NVIDIA GPU and driver plus CUDA 12 and cuDNN 9 native
libraries. CUDA 11, CUDA 13, and cuDNN 8 are not substitutes because
ctranslate2 loads the CUDA 12 and cuDNN 9 library names.

The supported setup on both platforms is:

```console
uv sync --extra cuda
uv run transcribe --debug
uv run transcribe --test
```

The CUDA extra installs the required libraries as wheels. On Windows, the tool
prepends their `bin` directories to its process `PATH` before importing
faster-whisper. Wheel libraries take precedence over `CUDA_PATH` and
`CUDNN_PATH`. A system installation can also work when the CUDA 12 and cuDNN 9
DLL directories are already on `PATH`, or those environment variables identify
the correct locations.

On Linux, the tool preloads the wheel libraries from their `lib` directories by
absolute path before importing faster-whisper. It does not guess system CUDA
locations. A system installation can work only when the required libraries are
already available to the dynamic loader.

Use `--debug` first. It reports the platform, candidate library directories,
and whether both required libraries were found without loading a model. Then
use `--test` for a real model, decoder, and device check.

## GPU failure behavior

- If GPU model construction fails at startup, report the underlying error,
  likely CUDA or NVIDIA GPU/driver causes, the CUDA extra installation command,
  and the deliberate `--cpu` alternative. Stop with exit code 1.
- `--cpu` skips GPU model construction and makes CPU use visible.
- If an already loaded GPU model fails on one file, retry that file on CPU.
- If the CPU retry succeeds, mark the GPU bad and use CPU for the rest of the
  run.
- If the CPU retry also fails, mark only that file failed. The next file still
  uses the GPU.

The retry and the switch of remaining files to CPU are both reported to stderr.
Preserve this state machine when changing model or error handling. A CPU retry
can be much slower, so do not treat slow transcription alone as a hang.

## Batch and output contract

- Skip an input only when both expected outputs exist. If one is missing,
  transcribe again and rewrite both.
- Match inputs to outputs by name, not content. Replacing media under the same
  filename requires `--overwrite` to replace existing transcripts.
- When two inputs in one directory share a stem, retain their extensions in
  output names. Recompute collisions on every run.
- Keep output as ASR text. Do not add speaker labels or present transcripts as
  reliable quotations.

## Exit codes

- `0`: all queued inputs succeeded, nothing was pending, or an informational
  action completed.
- `1`: the run could not start because arguments or inputs were invalid, the
  model name was unknown, or no model loaded before processing began.
- `2`: processing started and at least one queued file failed. Continue trying
  the other queued files before returning this code.

Preserve these meanings in CLI changes and tests.

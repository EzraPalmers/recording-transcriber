# recording-transcriber

Batch transcription of audio and video with faster-whisper. README.md covers
installation and the options a user needs; this file holds what an agent using
the tool needs and cannot read off the code.

This CLAUDE.md is committed and the repository is public, so nothing private
belongs in it: no absolute paths outside the repository, no tokens, no machine
or network detail.

## Layout

- `src/main.py`: CLI, CUDA path setup, orchestration.
- `src/transcribe.py`: file scan, output naming, the model wrapper.
- `tests/`: pytest for path and naming logic, GPU fallback, and model state.
  Tests use fakes; no real model is loaded.
- `.claude/skills/<name>/SKILL.md`: the single copy of each skill body. The
  matching file under `.agents/skills/` carries the same frontmatter for Codex
  and points here, so edit the `.claude/` copy and keep only the description in
  step with it.

`recordings/`, `transcripts/`, `transcripts_timestamps/`, `notes/`, and
`local/` are gitignored. Media and transcripts never enter a commit. Every one
of those except `local/` is created by the first `uv run transcribe`, so a
fresh clone has them after one run.

A skill description lists example phrasings, not the only way in. Load the
skill whose subject matches the work, however the request was worded.

## Running it

```console
uv sync                       # add --extra cuda for GPU
uv run transcribe --test      # one file, checks the setup end to end
uv run transcribe             # the batch
```

Every input under `recordings/` produces two files: plain text in
`transcripts/` and the same segments timestamped in `transcripts_timestamps/`,
both mirroring the subfolder layout.

Transcription is slow, minutes per recording even on a GPU, and a full batch is
a heavy run. Do not start one on your own initiative or in the background; hand
over the command instead. `--test` is the cheap check.

## Behaviour that surprises people

- An input is skipped only when BOTH of its outputs exist. If one is missing,
  it is transcribed again, which is what makes an interrupted batch resumable.
- Inputs are matched to outputs by name, not content. Replacing a recording
  under an existing filename leaves the old transcript in place until
  `--overwrite`.
- Two inputs in one folder that share a stem (`talk.mp3`, `talk.mp4`) both keep
  their extension in the output name, decided fresh on every run.
- Exit code 2 means at least one file failed; the rest of the batch still ran.
  Exit code 1 means the run could not start because its arguments or inputs were
  invalid, or because the model never loaded before processing began.
- Output is ASR text: no speaker labels, and technical terms and names are
  often misheard. Treat a transcript as a lead, not a quotation source.
- `--test` builds the batch from every input, then transcribes only the first
  pending one in sorted path order, so collision names match a normal run.
- `--suffix` rejects a value containing a path separator or `..`.
- The scanner accepts a fixed extension list, but decoding is PyAV's, so a
  listed file can still fail on its container or codec.

## Failure modes

If the GPU model cannot be built at startup, the run stops with exit code 1. The
error gives the underlying reason, likely CUDA or NVIDIA GPU/driver causes, the
CUDA extra installation command, and the deliberate `--cpu` alternative. A
loaded GPU model failing on one file is the only automatic CPU fallback. The
retry and any switch to CPU for the rest of the run are reported to stderr. CPU
success marks the GPU bad for the run; CPU failure marks only that file failed,
and the next file still uses the GPU.

The main trap is CUDA major version. ctranslate2 loads CUDA 12 and cuDNN 9
libraries by name, so CUDA 13 is not a substitute no matter how new the driver
is. Installing the wheels with `uv sync --extra cuda` is the fix. The tool finds
their `bin` directories on Windows or `lib` directories on Linux and exposes
the native libraries to ctranslate2 before importing faster-whisper. On
Windows, wheel libraries take precedence over `CUDA_PATH` and `CUDNN_PATH`.

`uv run transcribe --debug` reports the platform, discovered CUDA library
directories, and whether both required libraries were found, without loading a
model. Use it before blaming the model or the media file.

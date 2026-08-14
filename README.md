# recording-transcriber

This command-line tool batch-transcribes a folder of audio and video recordings
with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and it ships
with agent skills so that a coding agent can then work with the results. The
transcripts exist to be retrieved from rather than read: point an agent at them
and ask. `gather-notes` turns them into a sourced, timestamped note, and
`diagnose-gpu` works out why a run did not use the GPU. Claude Code loads the
skills from `.claude/skills/` and Codex from `.agents/skills/`, both without
setup, so ask in plain language rather than naming a file.

Each recording produces two transcripts, one plain and one with
`[hh:mm:ss -> hh:mm:ss]` in front of every segment, so a quote can be traced
back to its point in the audio. No speaker labels are written, so a recording
with one main talker suits it best.

## Installation

Install Python 3.10 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/). From the
repository root:

```console
uv sync --extra cuda
```

The `cuda` extra installs the CUDA 12 and cuDNN 9 libraries as wheels, which is
the supported way to use an NVIDIA GPU on Windows and Linux. Without an NVIDIA
GPU, install with plain `uv sync` and pass `--cpu` on every run.

The first transcription downloads the selected Whisper model from Hugging Face
and caches it outside the repository. No account is needed. A Hugging Face
token can speed that first download up: copy `.env.example` to `.env` and set
`HF_TOKEN`.

## Usage

Run the command once:

```console
uv run transcribe
```

On a fresh clone this creates the working folders, all gitignored, prints a
short guide to where files go, and exits. Put media anywhere under
`recordings/`, in whatever subfolders help you, and run the same command again
to transcribe everything pending. Both output trees mirror those subfolders.

Before a long batch, check the setup end to end on a single file:

```console
uv run transcribe --test
```

Accepted extensions are `.mp3`, `.mp4`, `.m4a`, `.wav`, `.flac`, `.ogg`,
`.opus`, `.mkv`, `.mov`, and `.webm`, in any case. Re-running after adding more
media transcribes only what is missing, so an interrupted batch is safe to
resume.

| Option | When to use it |
| --- | --- |
| `--cpu` | Bypass CUDA when no supported GPU is present, when GPU setup is still incomplete, or when reproducible CPU execution matters. CPU uses int8 computation. |
| `--model NAME` | Trade speed, memory use, and transcription quality by selecting another faster-whisper model. The default is `turbo`. |
| `--model-names` | Print the model names accepted by the installed faster-whisper version before choosing a value for `--model`. |
| `--overwrite` | Replace existing transcripts, for example after changing the model or replacing a recording under the same filename. Without it, a recording with both transcripts already written is skipped. |
| `--suffix TEXT` | Keep variants side by side by adding text to each output name, for example `--suffix _small`. |
| `--test` | Transcribe only the first pending file, to check the setup end to end before committing to a long batch. |
| `--debug` | Show the platform and the discovered CUDA 12 and cuDNN 9 library locations, then exit without loading a model. Use it to diagnose GPU setup. |

`uv run transcribe --help` prints the full option list.

Licensed under the MIT License. See [LICENSE](LICENSE).

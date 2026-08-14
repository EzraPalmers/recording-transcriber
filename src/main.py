"""Batch-transcribe audio and video recordings with faster-whisper.

Run commands:

    uv run transcribe                 transcribe pending files under recordings/
    uv run transcribe --cpu           transcribe on the CPU, needed without an NVIDIA GPU
    uv run transcribe --test          transcribe only the first pending file
    uv run transcribe --model small   use a specific Whisper model
    uv run transcribe --model-names   list the model names faster-whisper accepts
    uv run transcribe --debug         print platform CUDA diagnostics and exit

Files with both outputs already present are skipped unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import sys
import sysconfig
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv

WINDOWS_CUDA_LIBRARY_NAMES = ("cublas64_12.dll", "cudnn64_9.dll")
LINUX_CUDA_LIBRARY_NAMES = ("libcublas.so.12", "libcudnn.so.9")


def cuda_library_directories() -> list[Path]:
    """Native-library folders from the CUDA wheels or configured system installs."""
    site_packages = Path(sysconfig.get_paths()["purelib"])
    subdirectory = "bin" if platform.system() == "Windows" else "lib"
    wheel_directories = sorted(path for path in site_packages.glob(f"nvidia/*/{subdirectory}") if path.is_dir())
    if wheel_directories:
        return wheel_directories

    if platform.system() != "Windows":
        return []

    directories: list[Path] = []
    cuda_path = os.getenv("CUDA_PATH")
    cudnn_path = os.getenv("CUDNN_PATH")
    if cuda_path:
        directories.append(Path(cuda_path) / "bin")
    if cudnn_path:
        directories.append(Path(cudnn_path))
    return directories


def configure_cuda_libraries() -> None:
    directories = cuda_library_directories()
    if platform.system() == "Windows":
        parts = [*(str(path) for path in directories), os.environ.get("PATH", "")]
        os.environ["PATH"] = os.pathsep.join(part for part in parts if part)
        return

    if platform.system() == "Linux":
        for library_name in LINUX_CUDA_LIBRARY_NAMES:
            library_path = next((path / library_name for path in directories if (path / library_name).exists()), None)
            if library_path is not None:
                try:
                    ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


load_dotenv(override=True)
configure_cuda_libraries()

from faster_whisper import available_models  # noqa: E402 - must follow configure_cuda_libraries()

from src.transcribe import (  # noqa: E402 - same reason
    MEDIA_EXTENSIONS,
    ModelInitializationError,
    WhisperTranscriber,
    build_transcription_batch,
    scan_recordings,
    transcribe_batch,
)

DEFAULT_MODEL = "turbo"

RECORDINGS_ROOT = Path("recordings")
TRANSCRIPTS_ROOT = Path("transcripts")
TIMESTAMPS_ROOT = Path("transcripts_timestamps")
NOTES_ROOT = Path("notes")

WORKING_FOLDERS = (RECORDINGS_ROOT, TRANSCRIPTS_ROOT, TIMESTAMPS_ROOT, NOTES_ROOT)

GETTING_STARTED = f"""\
Put audio or video files anywhere under recordings/, then run 'transcribe'
again. Subfolders are allowed and the output mirrors them.

  recordings/               the media to transcribe
  transcripts/              plain text, one speech segment per line
  transcripts_timestamps/   the same segments, each marked [hh:mm:ss -> hh:mm:ss]
  notes/                    where an agent writes notes drawn from transcripts

Accepted file types: {" ".join(MEDIA_EXTENSIONS)}

  transcribe                transcribe everything without a transcript yet
  transcribe --test         transcribe one file first, to check the setup
  transcribe --cpu          transcribe on the CPU; needed on a machine with no
                            NVIDIA GPU, where a normal run stops at startup
  transcribe --overwrite    transcribe again, replacing existing transcripts
"""


class CommandLineParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _print_cuda_debug() -> None:
    system = platform.system()
    names = WINDOWS_CUDA_LIBRARY_NAMES if system == "Windows" else LINUX_CUDA_LIBRARY_NAMES
    directories = cuda_library_directories()
    print(f"Platform: {system}")
    print("CUDA library directories:")
    for directory in directories:
        print(f"  {directory}")
    if not directories:
        print("  none")

    for library_name in names:
        found = next((directory / library_name for directory in directories if (directory / library_name).exists()), None)
        if found:
            print(f"{library_name} found: {found}")
        else:
            print(f"{library_name} not found - GPU transcription cannot start")


def _suffix_value(value: str) -> str:
    if any(fragment in value for fragment in ("/", "\\", "..")):
        raise argparse.ArgumentTypeError("suffix must not contain a path separator or '..'")
    return value


def parse_args() -> argparse.Namespace:
    parser = CommandLineParser(
        description="Transcribe audio and video recordings with faster-whisper.",
        epilog=GETTING_STARTED,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Transcribe on the CPU. Required on a machine with no NVIDIA GPU, where a GPU run stops at startup.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcripts. By default, files with existing outputs are skipped.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        type=_suffix_value,
        help="Optional suffix appended to each transcript filename.",
    )
    parser.add_argument("--model-names", action="store_true", help="Print available model names and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--debug", action="store_true", help="Print platform CUDA library diagnostics, then exit.")
    parser.add_argument("--test", action="store_true", help="Run the pipeline on the first recording only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.debug:
        _print_cuda_debug()
        return 0

    model_names = list(available_models())

    if args.model_names:
        print("\n".join(model_names))
        return 0

    if args.model not in model_names:
        print(f"Unknown model: {args.model}", file=sys.stderr)
        print("Available models: " + ", ".join(model_names), file=sys.stderr)
        return 1

    folders_were_missing = not RECORDINGS_ROOT.exists()
    for folder in WORKING_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)

    recording_files = scan_recordings(RECORDINGS_ROOT)
    if not recording_files:
        if folders_were_missing:
            created = ", ".join(f"{folder}/" for folder in WORKING_FOLDERS)
            print(f"Created {created}.\n")
        else:
            print("Nothing to transcribe: no supported media found under recordings/.\n")
        print(GETTING_STARTED, end="")
        return 1

    batch = build_transcription_batch(
        recording_files=recording_files,
        recordings_root=RECORDINGS_ROOT,
        transcripts_root=TRANSCRIPTS_ROOT,
        timestamps_root=TIMESTAMPS_ROOT,
        overwrite=args.overwrite,
        suffix=args.suffix,
    )

    for group in batch.collision_groups:
        input_names = ", ".join(paths.input_path.name for paths in group)
        output_names = ", ".join(paths.output_path.name for paths in group)
        relative_folder = group[0].input_path.parent.relative_to(RECORDINGS_ROOT)
        folder_name = str(relative_folder) if str(relative_folder) != "." else "recordings/"
        print(
            f"{input_names} in {folder_name} would share one transcript name, "
            f"so their transcripts are named {output_names}.",
            file=sys.stderr,
        )

    pending_paths = batch.pending_paths
    if args.test and pending_paths:
        print(f"Test mode: running on {pending_paths[0].input_path.name} only.")
        pending_paths = pending_paths[:1]

    if not pending_paths:
        print(
            f"Detected {batch.total} recordings. "
            "All skipped because transcript outputs already exist. "
            "Use --overwrite to transcribe again."
        )
        return 0

    print(f"Detected {batch.total} recordings. Queued {len(pending_paths)} for transcription.")
    transcriber = WhisperTranscriber(
        model_size=args.model,
        use_gpu=not args.cpu,
        auth_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )
    try:
        transcriber.load_model()
        processed, failed = transcribe_batch(pending_paths, transcriber)
    except ModelInitializationError as exc:
        print(f"Could not initialise the Whisper model: {exc}", file=sys.stderr)
        return 1

    print(
        "Transcription complete. "
        f"processed={processed}, skipped={batch.skipped}, failed={failed}, total={batch.total}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

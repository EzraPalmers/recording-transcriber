from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from tqdm import tqdm

MEDIA_EXTENSIONS = (
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
    ".mkv",
    ".mov",
    ".webm",
)


class ModelInitializationError(RuntimeError):
    """Raised when the selected transcription model cannot be prepared."""


@dataclass(frozen=True)
class TranscriptionPaths:
    input_path: Path
    output_path: Path
    timestamps_path: Path


@dataclass(frozen=True)
class TranscriptionBatch:
    pending_paths: list[TranscriptionPaths]
    skipped: int
    total: int
    # Groups of inputs that share a stem, so each output keeps its extension to stay distinct.
    collision_groups: list[list[TranscriptionPaths]]


def scan_recordings(recordings_root: Path) -> list[Path]:
    """Find every supported media file under recordings_root, recursively."""
    return sorted(
        path for path in recordings_root.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def mirror_output_path(input_path: Path, recordings_root: Path, output_root: Path, stem: str, suffix: str = "") -> Path:
    """Map a recording to its output path, mirroring the folder layout under recordings_root."""
    relative_parent = input_path.relative_to(recordings_root).parent
    return output_root / relative_parent / f"{stem}{suffix}.txt"


def _output_stems(recording_files: list[Path]) -> list[str]:
    """Choose an output stem per file, keeping the extension wherever stems would clash."""
    stems = [path.stem for path in recording_files]
    while True:
        counts = Counter((path.parent, stem.casefold()) for path, stem in zip(recording_files, stems, strict=True))
        # A name can replace a stem, and that name can itself clash with another file's stem
        # (foo.mp3 against foo.mp3.wav), so repeat until the choice stops changing.
        updated = [
            path.name if counts[(path.parent, stem.casefold())] > 1 else stem
            for path, stem in zip(recording_files, stems, strict=True)
        ]
        if updated == stems:
            return stems
        stems = updated


def build_transcription_batch(
    recording_files: list[Path],
    recordings_root: Path,
    transcripts_root: Path,
    timestamps_root: Path,
    overwrite: bool,
    suffix: str,
) -> TranscriptionBatch:
    stems = _output_stems(recording_files)
    stem_counts = Counter((path.parent, path.stem.casefold()) for path in recording_files)

    pending_paths: list[TranscriptionPaths] = []
    skipped = 0
    collision_groups: dict[tuple[Path, str], list[TranscriptionPaths]] = {}

    for input_path, stem in zip(recording_files, stems, strict=True):
        paths = TranscriptionPaths(
            input_path=input_path,
            output_path=mirror_output_path(input_path, recordings_root, transcripts_root, stem, suffix),
            timestamps_path=mirror_output_path(input_path, recordings_root, timestamps_root, stem, suffix),
        )
        stem_key = (input_path.parent, input_path.stem.casefold())
        if stem_counts[stem_key] > 1:
            collision_groups.setdefault(stem_key, []).append(paths)

        if not overwrite and paths.output_path.exists() and paths.timestamps_path.exists():
            skipped += 1
            continue
        pending_paths.append(paths)

    return TranscriptionBatch(
        pending_paths=pending_paths,
        skipped=skipped,
        total=len(recording_files),
        collision_groups=list(collision_groups.values()),
    )


def format_timestamp(seconds: float) -> str:
    total_seconds = int(max(0, seconds))
    return f"{total_seconds // 3600:02d}:{total_seconds % 3600 // 60:02d}:{total_seconds % 60:02d}"


def write_transcripts(paths: TranscriptionPaths, segments: list[Any]) -> None:
    plain_lines: list[str] = []
    timed_lines: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        plain_lines.append(text)
        timed_lines.append(f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {text}")

    paths.output_path.parent.mkdir(parents=True, exist_ok=True)
    paths.timestamps_path.parent.mkdir(parents=True, exist_ok=True)
    paths.output_path.write_text("\n".join(plain_lines) + "\n", encoding="utf-8")
    paths.timestamps_path.write_text("\n".join(timed_lines) + "\n", encoding="utf-8")


class WhisperTranscriber:
    """Holds one loaded Whisper model, with per-file CPU retry after a GPU failure."""

    def __init__(self, model_size: str, use_gpu: bool = True, auth_token: str | None = None) -> None:
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.auth_token = auth_token
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        if self.use_gpu:
            try:
                self._model = self._build_model(device="cuda", compute_type="float16")
                return
            except Exception as gpu_error:
                raise ModelInitializationError(
                    f"GPU could not be initialised: {gpu_error}\n"
                    "Likely causes: the CUDA extra was not installed, so CUDA 12 and cuDNN 9 libraries are "
                    "missing, or no NVIDIA GPU or driver is available.\n"
                    "Install the CUDA libraries with 'uv sync --extra cuda', or re-run with --cpu for a "
                    "deliberate CPU run."
                ) from gpu_error
        tqdm.write("CPU transcription selected by --cpu.", file=sys.stderr)
        self._load_cpu_model()

    def transcribe_paths(self, paths: TranscriptionPaths) -> None:
        try:
            segments = self._transcribe(paths.input_path)
        except Exception as gpu_error:
            if not self.use_gpu:
                raise
            gpu_model = self._model
            tqdm.write(
                f"GPU transcription failed for {paths.input_path}: {gpu_error}. Retrying this file on CPU.",
                file=sys.stderr,
            )
            try:
                self._load_cpu_model()
                segments = self._transcribe(paths.input_path)
            except Exception as cpu_error:
                self._model = gpu_model
                raise RuntimeError(
                    f"GPU attempt failed: {gpu_error}. CPU retry failed: {cpu_error}"
                ) from cpu_error
            self._report_gpu_failure()
        write_transcripts(paths, segments)

    def _transcribe(self, input_path: Path) -> list[Any]:
        if self._model is None:
            raise ModelInitializationError("The model was not loaded before transcription started.")
        segments, _ = self._model.transcribe(str(input_path), vad_filter=True)
        return list(segments)

    def _build_model(self, device: str, compute_type: str) -> WhisperModel:
        # 75% of available threads on CPU, to leave the machine usable during a batch.
        cpu_threads = max(1, int((os.cpu_count() or 4) * 0.75)) if device == "cpu" else 0
        return WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            use_auth_token=self.auth_token,
        )

    def _load_cpu_model(self) -> None:
        try:
            self._model = self._build_model(device="cpu", compute_type="int8")
        except Exception as cpu_error:
            raise ModelInitializationError(str(cpu_error)) from cpu_error

    def _report_gpu_failure(self) -> None:
        self.use_gpu = False
        tqdm.write(
            "CPU retry succeeded. GPU disabled; using CPU for the remaining files in this run.",
            file=sys.stderr,
        )


def transcribe_batch(pending_paths: list[TranscriptionPaths], transcriber: WhisperTranscriber) -> tuple[int, int]:
    processed = 0
    failed = 0

    with tqdm(pending_paths, unit="file") as bar:
        for paths in bar:
            bar.set_description(paths.input_path.name)
            try:
                transcriber.transcribe_paths(paths)
                processed += 1
            except Exception as exc:
                tqdm.write(f"Transcription failed for {paths.input_path}: {exc}", file=sys.stderr)
                failed += 1

    return processed, failed

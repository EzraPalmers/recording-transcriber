from pathlib import Path

import pytest

from src.transcribe import (
    ModelInitializationError,
    TranscriptionPaths,
    WhisperTranscriber,
    build_transcription_batch,
    format_timestamp,
    mirror_output_path,
    scan_recordings,
)


class FakeModel:
    def __init__(self, outcomes: list[Exception | list[object]]) -> None:
        self.outcomes = outcomes

    def transcribe(self, _input_path: str, *, vad_filter: bool) -> tuple[list[object], None]:
        assert vad_filter is True
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome, None


def transcription_paths(tmp_path: Path, name: str = "input.mp3") -> TranscriptionPaths:
    return TranscriptionPaths(
        input_path=tmp_path / name,
        output_path=tmp_path / "transcripts" / f"{name}.txt",
        timestamps_path=tmp_path / "timestamps" / f"{name}.txt",
    )


def test_mirror_output_path_preserves_folders_and_applies_suffix(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    input_path = recordings / "team" / "meeting.mp3"

    assert mirror_output_path(input_path, recordings, tmp_path / "transcripts", "meeting", "_small") == (
        tmp_path / "transcripts" / "team" / "meeting_small.txt"
    )
    assert mirror_output_path(input_path, recordings, tmp_path / "transcripts", "meeting.mp3") == (
        tmp_path / "transcripts" / "team" / "meeting.mp3.txt"
    )


def test_build_transcription_batch_disambiguates_chained_names(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    files = [recordings / name for name in ("foo.mp3", "foo.mp4", "foo.mp3.wav")]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    batch = build_transcription_batch(
        files, recordings, tmp_path / "transcripts", tmp_path / "timestamps", False, ""
    )

    assert [path.output_path.name for path in batch.pending_paths] == [
        "foo.mp3.txt",
        "foo.mp4.txt",
        "foo.mp3.wav.txt",
    ]
    assert [[path.input_path.name for path in group] for group in batch.collision_groups] == [
        ["foo.mp3", "foo.mp4"]
    ]


def test_build_transcription_batch_skips_complete_pairs_and_retries_partial_pair(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    transcripts = tmp_path / "transcripts"
    timestamps = tmp_path / "timestamps"
    complete = recordings / "complete.mp3"
    partial = recordings / "partial.wav"
    recordings.mkdir(parents=True)
    complete.touch()
    partial.touch()

    (transcripts / "complete.txt").parent.mkdir(parents=True, exist_ok=True)
    (transcripts / "complete.txt").touch()
    (timestamps / "complete.txt").parent.mkdir(parents=True, exist_ok=True)
    (timestamps / "complete.txt").touch()
    (transcripts / "partial.txt").touch()

    batch = build_transcription_batch(
        [complete, partial], recordings, transcripts, timestamps, False, ""
    )

    assert batch.skipped == 1
    assert [path.input_path.name for path in batch.pending_paths] == ["partial.wav"]
    assert batch.pending_paths[0].output_path == transcripts / "partial.txt"


def test_scan_recordings_filters_extensions_case_insensitively_and_recurses(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    supported = [recordings / "root.MP3", recordings / "nested" / "clip.WaV"]
    for path in supported:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (recordings / "nested" / "notes.txt").touch()

    assert scan_recordings(recordings) == sorted(supported)


def test_gpu_success_keeps_gpu_for_next_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_model = FakeModel([[], []])
    transcriber = WhisperTranscriber("turbo")
    transcriber._model = gpu_model
    monkeypatch.setattr("src.transcribe.write_transcripts", lambda _paths, _segments: None)

    transcriber.transcribe_paths(transcription_paths(tmp_path, "one.mp3"))
    transcriber.transcribe_paths(transcription_paths(tmp_path, "two.mp3"))

    assert transcriber.use_gpu is True
    assert transcriber._model is gpu_model


def test_gpu_load_failure_stops_without_building_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    model_builds: list[tuple[str, str]] = []
    transcriber = WhisperTranscriber("turbo")

    def build_model(*, device: str, compute_type: str) -> FakeModel:
        model_builds.append((device, compute_type))
        raise RuntimeError("gpu load failed")

    monkeypatch.setattr(transcriber, "_build_model", build_model)

    with pytest.raises(ModelInitializationError) as exc_info:
        transcriber.load_model()

    message = str(exc_info.value)
    assert "GPU could not be initialised: gpu load failed" in message
    assert "CUDA extra was not installed" in message
    assert "CUDA 12 and cuDNN 9 libraries are missing" in message
    assert "no NVIDIA GPU or driver" in message
    assert "uv sync --extra cuda" in message
    assert "re-run with --cpu" in message
    assert model_builds == [("cuda", "float16")]
    assert transcriber.use_gpu is True
    assert transcriber._model is None


def test_explicit_cpu_skips_gpu_initialisation_and_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cpu_model = FakeModel([[]])
    model_builds: list[tuple[str, str]] = []
    transcriber = WhisperTranscriber("turbo", use_gpu=False)

    def build_model(*, device: str, compute_type: str) -> FakeModel:
        model_builds.append((device, compute_type))
        return cpu_model

    monkeypatch.setattr(transcriber, "_build_model", build_model)

    transcriber.load_model()

    captured = capsys.readouterr()
    assert model_builds == [("cpu", "int8")]
    assert transcriber._model is cpu_model
    assert captured.out == ""
    assert "CPU transcription selected by --cpu." in captured.err


def test_gpu_failure_then_cpu_success_switches_remaining_files_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gpu_model = FakeModel([RuntimeError("gpu failed")])
    cpu_model = FakeModel([[], []])
    transcriber = WhisperTranscriber("turbo")
    transcriber._model = gpu_model
    monkeypatch.setattr(transcriber, "_build_model", lambda **_kwargs: cpu_model)
    monkeypatch.setattr("src.transcribe.write_transcripts", lambda _paths, _segments: None)

    transcriber.transcribe_paths(transcription_paths(tmp_path, "one.mp3"))
    transcriber.transcribe_paths(transcription_paths(tmp_path, "two.mp3"))

    captured = capsys.readouterr()
    assert transcriber.use_gpu is False
    assert transcriber._model is cpu_model
    assert captured.out == ""
    assert "GPU transcription failed for" in captured.err
    assert "Retrying this file on CPU." in captured.err
    assert "CPU retry succeeded. GPU disabled; using CPU for the remaining files" in captured.err


def test_gpu_and_cpu_failure_leaves_gpu_enabled_for_next_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_model = FakeModel([RuntimeError("gpu failed"), []])
    cpu_model = FakeModel([RuntimeError("cpu failed")])
    transcriber = WhisperTranscriber("turbo")
    transcriber._model = gpu_model
    monkeypatch.setattr(transcriber, "_build_model", lambda **_kwargs: cpu_model)
    monkeypatch.setattr("src.transcribe.write_transcripts", lambda _paths, _segments: None)

    with pytest.raises(RuntimeError, match="GPU attempt failed: gpu failed. CPU retry failed: cpu failed"):
        transcriber.transcribe_paths(transcription_paths(tmp_path, "bad.mp3"))
    transcriber.transcribe_paths(transcription_paths(tmp_path, "good.mp3"))

    assert transcriber.use_gpu is True
    assert transcriber._model is gpu_model


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00:00"), (-2, "00:00:00"), (61.9, "00:01:01"), (3661, "01:01:01")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected

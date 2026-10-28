import time
import asyncio
import tempfile
import threading
from pathlib import Path
from typing import List, Dict, Optional
import yt_dlp
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from litellm import transcription as litellm_transcription
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from ..models import ContentSegment, VideoSnapshot
from ..cache import PipelineCache
from .visual_utils import VideoSnapshotExtractor
from .ebook import EbookIngestor

console = Console()


class WhisperAudioTranscriber:
    """Thread-safe local faster-whisper / cloud API audio transcriber."""

    def __init__(
        self,
        mode: str = "local",
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        api_model: str = "groq/whisper-large-v3-turbo",
        batch_size: int = 8,
    ):
        self.mode = mode.lower()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.api_model = api_model
        self.batch_size = batch_size

        self._local_model = None
        self._batched_pipeline = None
        self._model_lock = threading.Lock()

    def _ensure_local_model_loaded(self):
        if self._local_model is not None:
            return
        import ctranslate2
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        resolved_device = self.device
        if resolved_device == "auto":
            try:
                resolved_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                resolved_device = "cpu"

        resolved_compute = self.compute_type
        if resolved_compute == "auto":
            resolved_compute = "float16" if resolved_device == "cuda" else "int8"

        console.print(
            f"[cyan]Loading local faster-whisper '{self.model_size}' "
            f"(device={resolved_device}, compute={resolved_compute})...[/cyan]"
        )
        self._local_model = WhisperModel(self.model_size, device=resolved_device, compute_type=resolved_compute)
        try:
            self._batched_pipeline = BatchedInferencePipeline(model=self._local_model)
        except Exception:
            self._batched_pipeline = None

    @staticmethod
    def _download_audio_stream(video_id: str, temp_dir: str) -> Path:
        video_url = f"[https://www.youtube.com/watch?v=](https://www.youtube.com/watch?v=){video_id}"
        out_template = str(Path(temp_dir) / f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio[ext=m4a][abr<=64]/bestaudio[abr<=64]/bestaudio[ext=m4a]/bestaudio",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        files = list(Path(temp_dir).glob(f"{video_id}.*"))
        if not files:
            raise FileNotFoundError(f"Audio stream download failed for {video_id}")
        return files[0]

    def transcribe_video(self, video_id: str, preferred_language: Optional[str] = None) -> str:
        with tempfile.TemporaryDirectory(prefix="skillforge_audio_") as tmp_dir:
            audio_path = self._download_audio_stream(video_id, tmp_dir)

            if self.mode == "api":
                with open(audio_path, "rb") as audio_file:
                    kwargs = {"model": self.api_model, "file": audio_file}
                    if preferred_language:
                        kwargs["language"] = preferred_language
                    response = litellm_transcription(**kwargs)
                    text = response.get("text", "") if isinstance(response, dict) else getattr(response, "text", "")
                    return text.strip()

            with self._model_lock:
                self._ensure_local_model_loaded()
                transcribe_kwargs = {
                    "vad_filter": True,
                    "vad_parameters": {"min_silence_duration_ms": 500},
                    "beam_size": 5,
                }
                if preferred_language:
                    transcribe_kwargs["language"] = preferred_language

                if self._batched_pipeline is not None and self.batch_size > 1:
                    segments, _ = self._batched_pipeline.transcribe(
                        str(audio_path), batch_size=self.batch_size, **transcribe_kwargs
                    )
                else:
                    segments, _ = self._local_model.transcribe(str(audio_path), **transcribe_kwargs)

                return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()


class YouTubeIngestor:
    """Concurrently ingests YouTube videos/playlists with captions, Whisper fallback, and visual snapshots."""

    @staticmethod
    def _process_single_video_sync(
        vid: Dict[str, str],
        cache: PipelineCache,
        languages: List[str],
        request_delay: float,
        enable_whisper_fallback: bool,
        whisper_transcriber: Optional[WhisperAudioTranscriber],
        enable_visuals: bool,
        snapshot_extractor: Optional[VideoSnapshotExtractor],
    ) -> Optional[ContentSegment]:
        vid_id = vid["id"]
        vid_title = vid["title"]

        full_text = cache.get_transcript(vid_id)
        if not full_text:
            ytt_api = YouTubeTranscriptApi()
            try:
                try:
                    transcript = ytt_api.fetch(vid_id, languages=languages)
                except NoTranscriptFound:
                    transcript_list = ytt_api.list(vid_id)
                    first_available = next(iter(transcript_list))
                    transcript = first_available.fetch()

                full_text = " ".join(snippet.text for snippet in transcript).strip()
                if request_delay > 0:
                    time.sleep(request_delay)
            except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, StopIteration, Exception) as cap_err:
                if not enable_whisper_fallback or whisper_transcriber is None:
                    console.print(f"[yellow]Skipping '{vid_title}' ({vid_id}) - no captions ({cap_err}).[/yellow]")
                    return None
                try:
                    full_text = whisper_transcriber.transcribe_video(vid_id)
                except Exception as w_err:
                    console.print(f"[red]Whisper fallback failed for '{vid_title}' ({vid_id}): {w_err}[/red]")
                    return None

            if full_text and len(full_text) > 200:
                cache.save_transcript(vid_id, vid_title, full_text)
            else:
                return None

        snapshots: List[VideoSnapshot] = []
        if enable_visuals and snapshot_extractor is not None:
            cached_snaps = cache.get_snapshots(f"yt_{vid_id}")
            if cached_snaps is not None:
                snapshots = cached_snaps
            else:
                try:
                    snapshots = snapshot_extractor.extract_snapshots(vid_id)
                    cache.save_snapshots(f"yt_{vid_id}", snapshots)
                except Exception as snap_err:
                    console.print(f"[yellow]Snapshot extraction skipped for '{vid_title}': {snap_err}[/yellow]")

        return ContentSegment(
            segment_id=f"yt_{vid_id}",
            title=vid_title,
            text=full_text,
            snapshots=snapshots,
        )

    @staticmethod
    async def aload(
        url: str,
        cache: PipelineCache,
        ingest_concurrency: int = 4,
        languages: Optional[List[str]] = None,
        request_delay: float = 0.5,
        max_chars_per_segment: int = 40000,
        enable_whisper_fallback: bool = True,
        whisper_transcriber: Optional[WhisperAudioTranscriber] = None,
        enable_visuals: bool = True,
        snapshot_extractor: Optional[VideoSnapshotExtractor] = None,
    ) -> List[ContentSegment]:
        if languages is None:
            languages = ["en", "ar", "es", "fr", "de", "pt", "hi", "zh-Hans"]
        if enable_whisper_fallback and whisper_transcriber is None:
            whisper_transcriber = WhisperAudioTranscriber()
        if enable_visuals and snapshot_extractor is None:
            snapshot_extractor = VideoSnapshotExtractor()

        ydl_opts = {
            "extract_flat": "in_playlist",
            "quiet": True,
            "skip_download": True,
            "ignoreerrors": True,
        }
        info = await asyncio.to_thread(lambda: yt_dlp.YoutubeDL(ydl_opts).extract_info(url, download=False))
        if not info:
            raise ValueError(f"Could not retrieve metadata for URL: {url}")

        videos = []
        if "entries" in info:
            for entry in info["entries"]:
                if entry and entry.get("id"):
                    videos.append({"id": entry["id"], "title": entry.get("title") or entry["id"]})
        else:
            videos.append({"id": info["id"], "title": info.get("title") or info["id"]})

        console.print(f"Discovered [bold cyan]{len(videos)}[/bold cyan] video(s). Ingesting with concurrency={ingest_concurrency}...")
        ingest_sem = asyncio.Semaphore(ingest_concurrency)
        ordered_results: List[Optional[ContentSegment]] = [None] * len(videos)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Ingesting transcripts & visual snapshots...", total=len(videos))

            async def _worker(idx: int, vid_meta: Dict[str, str]):
                async with ingest_sem:
                    seg = await asyncio.to_thread(
                        YouTubeIngestor._process_single_video_sync,
                        vid_meta,
                        cache,
                        languages,
                        request_delay,
                        enable_whisper_fallback,
                        whisper_transcriber,
                        enable_visuals,
                        snapshot_extractor,
                    )
                    ordered_results[idx] = seg
                    progress.advance(task_id)

            await asyncio.gather(*[_worker(i, v) for i, v in enumerate(videos)])

        valid_segments = [s for s in ordered_results if s is not None]
        return EbookIngestor.normalize_segments(valid_segments, max_chars_per_segment)

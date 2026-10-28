import io
import base64
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yt_dlp
from ..models import VideoSnapshot


def compute_visual_info_score(gray_img: np.ndarray) -> float:
    """
    Measures Canny edge density + contrast variance.
    Charts, tables, diagrams, and equations score higher than blank backgrounds or solid logos.
    """
    if gray_img.size == 0:
        return 0.0
    edges = cv2.Canny(gray_img, 80, 180)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size) * 100.0
    contrast_std = float(np.std(gray_img)) / 255.0 * 10.0
    return round(edge_ratio + contrast_std, 3)


def encode_pil_image_with_badge(
    pil_img: Image.Image,
    badge_text: str,
    max_dimension: int = 1024,
    jpeg_quality: int = 80,
) -> Tuple[str, float]:
    """
    Converts a PIL image to RGB, computes its visual information density score,
    resizes to `max_dimension`, stamps a top-left `[badge_text]` label, and returns (base64_jpeg, info_score).
    """
    rgb_img = pil_img.convert("RGB")
    np_rgb = np.array(rgb_img)
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    info_score = compute_visual_info_score(gray)

    w, h = rgb_img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        rgb_img = rgb_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(rgb_img)
    label = f" [{badge_text}] "
    box_width = max(110, len(label) * 8)
    draw.rectangle([(8, 8), (box_width, 34)], fill=(0, 0, 0))
    draw.text((14, 13), label, fill=(255, 255, 0))

    buf = io.BytesIO()
    rgb_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64_str, info_score


def select_top_snapshots(candidates: List[VideoSnapshot], max_snapshots: int) -> List[VideoSnapshot]:
    """Selects top K snapshots by info_score and restores chronological/page order."""
    if len(candidates) <= max_snapshots:
        return sorted(candidates, key=lambda x: x.timestamp_sec)
    top_k = sorted(candidates, key=lambda x: x.info_score, reverse=True)[:max_snapshots]
    return sorted(top_k, key=lambda x: x.timestamp_sec)


class VideoSnapshotExtractor:
    """Extracts deduplicated, high-information keyframes from YouTube videos."""

    def __init__(
        self,
        max_snapshots: int = 12,
        sample_interval_sec: float = 8.0,
        min_scene_diff_threshold: float = 12.0,
        max_image_dimension: int = 1024,
        jpeg_quality: int = 80,
    ):
        self.max_snapshots = max_snapshots
        self.sample_interval_sec = sample_interval_sec
        self.min_scene_diff_threshold = min_scene_diff_threshold
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        mins, secs = int(seconds // 60), int(seconds % 60)
        hours, mins = int(mins // 60), mins % 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"

    @staticmethod
    def _download_compact_video(video_id: str, temp_dir: str) -> Path:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        out_template = str(Path(temp_dir) / f"{video_id}_vid.%(ext)s")
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/bestvideo[height<=480]/best[height<=480]",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        files = list(Path(temp_dir).glob(f"{video_id}_vid.*"))
        if not files:
            raise FileNotFoundError(f"Video stream download failed for {video_id}")
        return files[0]

    def extract_snapshots(self, video_id: str) -> List[VideoSnapshot]:
        with tempfile.TemporaryDirectory(prefix="skillforge_vid_") as tmp_dir:
            video_path = self._download_compact_video(video_id, tmp_dir)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration_sec = total_frames / fps if total_frames > 0 else 0
            interval = (
                max(self.sample_interval_sec, duration_sec / 120.0)
                if duration_sec > 0
                else self.sample_interval_sec
            )

            candidates: List[VideoSnapshot] = []
            prev_small_gray: Optional[np.ndarray] = None
            current_sec = 2.0

            while True:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000.0)
                success, frame = cap.read()
                if not success or frame is None:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray, (64, 36))
                mean_lum = float(np.mean(small_gray))

                if 15.0 < mean_lum < 245.0:
                    is_distinct = True
                    if prev_small_gray is not None:
                        diff = float(np.mean(cv2.absdiff(small_gray, prev_small_gray)))
                        is_distinct = diff >= self.min_scene_diff_threshold

                    if is_distinct:
                        ts_str = self._format_timestamp(current_sec)
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        b64_jpg, info_score = encode_pil_image_with_badge(
                            pil_img,
                            badge_text=ts_str,
                            max_dimension=self.max_image_dimension,
                            jpeg_quality=self.jpeg_quality,
                        )
                        candidates.append(
                            VideoSnapshot(
                                timestamp_sec=round(current_sec, 1),
                                timestamp_str=ts_str,
                                base64_jpeg=b64_jpg,
                                info_score=info_score,
                            )
                        )
                        prev_small_gray = small_gray

                current_sec += interval
                if duration_sec > 0 and current_sec >= duration_sec:
                    break

            cap.release()
            return select_top_snapshots(candidates, self.max_snapshots)

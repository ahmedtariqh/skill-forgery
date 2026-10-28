import io
import posixpath
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from PIL import Image

from ..models import ContentSegment, VideoSnapshot
from ..cache import PipelineCache
from .visual_utils import encode_pil_image_with_badge, select_top_snapshots


class EbookIngestor:
    """
    Extracts structured chapters/sections AND embedded figures, charts, diagrams,
    and vector-drawn tables from PDF and EPUB ebooks.
    """

    MIN_IMAGE_WIDTH = 160
    MIN_IMAGE_HEIGHT = 160
    MIN_INFO_SCORE = 1.8

    @staticmethod
    def load(
        file_path: str,
        cache: Optional[PipelineCache] = None,
        max_chars_per_chunk: int = 35000,
        enable_visuals: bool = True,
        max_snapshots_per_chapter: int = 10,
    ) -> List[ContentSegment]:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            raw_segments = EbookIngestor._read_pdf(
                path,
                cache=cache,
                enable_visuals=enable_visuals,
                max_snapshots_per_chapter=max_snapshots_per_chapter,
            )
        elif ext == ".epub":
            raw_segments = EbookIngestor._read_epub(
                path,
                cache=cache,
                enable_visuals=enable_visuals,
                max_snapshots_per_chapter=max_snapshots_per_chapter,
            )
        else:
            raise ValueError(f"Unsupported ebook format: {ext}. Use .pdf or .epub.")

        return EbookIngestor.normalize_segments(raw_segments, max_chars_per_chunk)

    @staticmethod
    def _extract_pdf_page_visuals(doc: fitz.Document, page_index: int) -> List[VideoSnapshot]:
        """
        Extracts both:
          1. Embedded raster images (`page.get_images()`)
          2. Vector-drawn charts, flowcharts, and bordered tables (`page.get_drawings()`)
        from a single PDF page.
        """
        page = doc[page_index]
        page_num = page_index + 1
        snapshots: List[VideoSnapshot] = []
        seen_xrefs = set()

        # 1. Extract embedded raster figures/charts
        for img_idx, img_info in enumerate(page.get_images(full=True), 1):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base_img = doc.extract_image(xref)
                if not base_img or "image" not in base_img:
                    continue
                pil_img = Image.open(io.BytesIO(base_img["image"]))
                w, h = pil_img.size
                if w < EbookIngestor.MIN_IMAGE_WIDTH or h < EbookIngestor.MIN_IMAGE_HEIGHT:
                    continue
                aspect = max(w, h) / float(max(1, min(w, h)))
                if aspect > 7.0:  # Skip thin decorative page rules/banners
                    continue

                badge = f"Page {page_num} Fig.{img_idx}"
                b64_jpg, score = encode_pil_image_with_badge(pil_img, badge_text=badge)
                if score >= EbookIngestor.MIN_INFO_SCORE:
                    snapshots.append(
                        VideoSnapshot(
                            timestamp_sec=float(page_num) + (img_idx * 0.01),
                            timestamp_str=badge,
                            base64_jpeg=b64_jpg,
                            info_score=score,
                        )
                    )
            except Exception:
                continue

        # 2. Detect vector-drawn charts, anatomical/financial diagrams, or complex tables
        # If no raster figure was found on this page, check if the page has rich vector drawings
        if not snapshots:
            try:
                drawings = page.get_drawings()
                rects = [
                    fitz.Rect(d["rect"])
                    for d in drawings
                    if d.get("rect") and d["rect"].width > 4 and d["rect"].height > 4
                ]
                if len(rects) >= 18:
                    union_rect = rects[0]
                    for r in rects[1:]:
                        union_rect |= r
                    # Expand slightly to include axis labels and legends
                    union_rect = union_rect + (-15, -15, 15, 15)
                    union_rect &= page.rect

                    if union_rect.width >= 150 and union_rect.height >= 120:
                        pix = page.get_pixmap(clip=union_rect, dpi=160)
                        pil_vec = Image.open(io.BytesIO(pix.tobytes("png")))
                        badge = f"Page {page_num} Chart"
                        b64_jpg, score = encode_pil_image_with_badge(pil_vec, badge_text=badge)
                        if score >= EbookIngestor.MIN_INFO_SCORE + 0.5:
                            snapshots.append(
                                VideoSnapshot(
                                    timestamp_sec=float(page_num),
                                    timestamp_str=badge,
                                    base64_jpeg=b64_jpg,
                                    info_score=score,
                                )
                            )
            except Exception:
                pass

        return snapshots

    @staticmethod
    def _read_pdf(
        path: Path,
        cache: Optional[PipelineCache],
        enable_visuals: bool,
        max_snapshots_per_chapter: int,
    ) -> List[ContentSegment]:
        doc = fitz.open(path)
        toc = doc.get_toc()
        segments: List[ContentSegment] = []

        # Build chapter page ranges: (chapter_id, title, start_page_0_idx, end_page_0_idx)
        ranges = []
        if toc:
            for i, entry in enumerate(toc):
                _, title, start_page = entry
                end_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(doc)
                s_idx = max(0, start_page - 1)
                e_idx = min(len(doc), max(s_idx + 1, end_page))
                ranges.append((f"{path.stem}_ch_{i+1}", title, s_idx, e_idx))
        else:
            pages_per_group = 10
            for start in range(0, len(doc), pages_per_group):
                end = min(start + pages_per_group, len(doc))
                ranges.append((f"{path.stem}_p_{start+1}_{end}", f"Pages {start+1}-{end}", start, end))

        for seg_id, title, s_idx, e_idx in ranges:
            text_pages = [doc[p].get_text("text") for p in range(s_idx, e_idx)]
            combined_text = "\n".join(text_pages).strip()
            if len(combined_text) <= 250:
                continue

            chapter_snapshots: List[VideoSnapshot] = []
            if enable_visuals:
                cached_snaps = cache.get_snapshots(seg_id) if cache else None
                if cached_snaps is not None:
                    chapter_snapshots = cached_snaps
                else:
                    raw_snaps: List[VideoSnapshot] = []
                    for p in range(s_idx, e_idx):
                        raw_snaps.extend(EbookIngestor._extract_pdf_page_visuals(doc, p))
                    chapter_snapshots = select_top_snapshots(raw_snaps, max_snapshots_per_chapter)
                    if cache:
                        cache.save_snapshots(seg_id, chapter_snapshots)

            segments.append(
                ContentSegment(
                    segment_id=seg_id,
                    title=title,
                    text=combined_text,
                    snapshots=chapter_snapshots,
                )
            )

        doc.close()
        return segments

    @staticmethod
    def _read_epub(
        path: Path,
        cache: Optional[PipelineCache],
        enable_visuals: bool,
        max_snapshots_per_chapter: int,
    ) -> List[ContentSegment]:
        book = epub.read_epub(str(path))
        segments: List[ContentSegment] = []

        # Map image filenames/paths inside the EPUB archive to their raw binary content
        image_lookup: Dict[str, bytes] = {}
        if enable_visuals:
            for item in book.get_items():
                media_type = (item.media_type or "").lower()
                if item.get_type() == ebooklib.ITEM_IMAGE or media_type.startswith("image/"):
                    norm_name = posixpath.normpath(item.get_name()).lstrip("/")
                    image_lookup[norm_name] = item.get_content()
                    image_lookup[posixpath.basename(norm_name)] = item.get_content()

        for idx, doc_item in enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), 1):
            soup = BeautifulSoup(doc_item.get_content(), "html.parser")
            heading = soup.find(["h1", "h2", "h3", "title"])
            title = heading.get_text(strip=True) if heading else doc_item.get_name()
            text = soup.get_text(separator="\n", strip=True)
            if len(text) <= 350:
                continue

            seg_id = f"{path.stem}_sec_{idx}"
            chapter_snapshots: List[VideoSnapshot] = []

            if enable_visuals:
                cached_snaps = cache.get_snapshots(seg_id) if cache else None
                if cached_snaps is not None:
                    chapter_snapshots = cached_snaps
                else:
                    doc_dir = posixpath.dirname(doc_item.get_name())
                    img_tags = soup.find_all(["img", "image"])
                    raw_snaps: List[VideoSnapshot] = []

                    for fig_idx, tag in enumerate(img_tags, 1):
                        src = tag.get("src") or tag.get("xlink:href") or tag.get("href")
                        if not src:
                            continue
                        src_clean = src.split("#")[0].split("?")[0]
                        rel_resolved = posixpath.normpath(posixpath.join(doc_dir, src_clean)).lstrip("/")
                        img_bytes = (
                            image_lookup.get(rel_resolved)
                            or image_lookup.get(src_clean.lstrip("/"))
                            or image_lookup.get(posixpath.basename(src_clean))
                        )
                        if not img_bytes:
                            continue

                        try:
                            pil_img = Image.open(io.BytesIO(img_bytes))
                            w, h = pil_img.size
                            if w < EbookIngestor.MIN_IMAGE_WIDTH or h < EbookIngestor.MIN_IMAGE_HEIGHT:
                                continue
                            badge = f"Ch.{idx} Fig.{fig_idx}"
                            b64_jpg, score = encode_pil_image_with_badge(pil_img, badge_text=badge)
                            if score >= EbookIngestor.MIN_INFO_SCORE:
                                raw_snaps.append(
                                    VideoSnapshot(
                                        timestamp_sec=float(idx * 100 + fig_idx),
                                        timestamp_str=badge,
                                        base64_jpeg=b64_jpg,
                                        info_score=score,
                                    )
                                )
                        except Exception:
                            continue

                    chapter_snapshots = select_top_snapshots(raw_snaps, max_snapshots_per_chapter)
                    if cache:
                        cache.save_snapshots(seg_id, chapter_snapshots)

            segments.append(
                ContentSegment(
                    segment_id=seg_id,
                    title=title,
                    text=text,
                    snapshots=chapter_snapshots,
                )
            )

        return segments

    @staticmethod
    def normalize_segments(segments: List[ContentSegment], max_chars: int) -> List[ContentSegment]:
        normalized: List[ContentSegment] = []
        for seg in segments:
            if len(seg.text) <= max_chars:
                normalized.append(seg)
            else:
                paragraphs = seg.text.split("\n\n")
                buffer = ""
                chunks_text = []
                for para in paragraphs:
                    if len(buffer) + len(para) > max_chars and buffer:
                        chunks_text.append(buffer)
                        buffer = para
                    else:
                        buffer = f"{buffer}\n\n{para}" if buffer else para
                if buffer:
                    chunks_text.append(buffer)

                total_parts = len(chunks_text)
                snaps = seg.snapshots
                for p_idx, c_text in enumerate(chunks_text):
                    start_s = int(p_idx * len(snaps) / total_parts)
                    end_s = int((p_idx + 1) * len(snaps) / total_parts)
                    normalized.append(
                        ContentSegment(
                            segment_id=f"{seg.segment_id}_pt{p_idx+1}",
                            title=f"{seg.title} (Part {p_idx+1})",
                            text=c_text,
                            snapshots=snaps[start_s:end_s],
                        )
                    )
        return normalized

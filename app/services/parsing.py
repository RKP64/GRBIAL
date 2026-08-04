from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
from pypdf import PdfReader

SUPPORTED = {".pdf", ".xlsx", ".xls", ".csv", ".txt", ".md"}


@dataclass
class Chunk:
    index: int
    source: str
    text: str


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware splitter — no LangChain dependency."""
    text = text.strip()
    if not text:
        return []
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= size:
                buf = para
            else:
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + size])
                    start += max(1, size - overlap)
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def chunk_file(filename: str, data: bytes, *, rows_per_chunk: int, chunk_size: int, overlap: int) -> list[Chunk]:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED)}")

    out: list[Chunk] = []
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        out = [Chunk(i, filename, t) for i, t in enumerate(_split(text, chunk_size, overlap))]

    elif ext in {".xlsx", ".xls", ".csv"}:
        frames: list[tuple[str, pd.DataFrame]] = []
        if ext == ".csv":
            frames.append(("csv", pd.read_csv(io.BytesIO(data))))
        else:
            book = pd.ExcelFile(io.BytesIO(data))
            for sheet in book.sheet_names:
                frames.append((sheet, pd.read_excel(book, sheet_name=sheet)))
        i = 0
        for sheet, df in frames:
            df = df.dropna(how="all")
            if df.empty:
                continue
            for start in range(0, len(df), rows_per_chunk):
                sub = df.iloc[start : start + rows_per_chunk]
                body = sub.to_csv(index=False)
                out.append(
                    Chunk(i, f"{filename}#{sheet}", f"Sheet: {sheet} | rows {start+1}-{start+len(sub)}\n{body}")
                )
                i += 1
    else:
        text = data.decode("utf-8", errors="replace")
        out = [Chunk(i, filename, t) for i, t in enumerate(_split(text, chunk_size, overlap))]

    return out

import re


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """
    Simple character-based chunker. It prefers paragraph boundaries, then sentence boundaries.
    """
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
            continue

        if current:
            chunks.append(current)

        if len(para) <= chunk_size:
            current = para
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sentence in sentences:
            if len(buf) + len(sentence) + 1 <= chunk_size:
                buf = f"{buf} {sentence}".strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = sentence
        current = buf

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prefix = chunks[i - 1][-overlap:]
        overlapped.append(f"{prefix}\n\n{chunks[i]}")
    return overlapped

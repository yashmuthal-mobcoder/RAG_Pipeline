import re
import uuid
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def detect_noise_lines(text: str, repeat_threshold: int = 3) -> set[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = Counter(lines)
    return {line for line, count in counts.items() if count >= repeat_threshold}

def clean_pdf_text(text: str, extra_noise: set[str] | None = None) -> str:
    noise_lines = detect_noise_lines(text)
    if extra_noise:
        noise_lines |= extra_noise

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in noise_lines or re.fullmatch(r"\d{1,4}", stripped):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = _fix_ocr_splits(text)

    return text.strip()

_OCR_FIXES: dict[str, str] = {
    r"INTRODUC\s+TION": "INTRODUCTION",
    r"DONOR\s+REQUES\s+TS\s+FOR\s+DATA": "DONOR REQUESTS FOR DATA",
    r"OBJEC\s+TIVES\s+FOR\s+DATA\s+SHARING\s+WITH\s+DONOR\s*S":
        "OBJECTIVES FOR DATA SHARING WITH DONORS",
    r"CONS\s*TR\s*AINTS\s+FOR\s+DATA\s+SHARING\s+WITH\s+DONOR\s*S":
        "CONSTRAINTS FOR DATA SHARING WITH DONORS",
    r"DATA\s+RESPONSIBILIT\s*Y\s+IN\s+HUMANITARIAN\s+AC\s*TION":
        "DATA RESPONSIBILITY IN HUMANITARIAN ACTION",
    r"KEY\s+TAKE\s*AWAYS\s*:?": "KEY TAKEAWAYS",
    r"THE\s+CENTRE\s+FOR\s+HUM\s*ANITARIAN\s+DATA":
        "THE CENTRE FOR HUMANITARIAN DATA",
    r"\b2020\s*\n\s*,": "2020,",
    r"\b2019\s*\n\s*and\s*\n\s*2020": "2019 and 2020",
    r"\bMarch\s*\n\s*2019": "March 2019",
    r"\bSeptember\s+,": "September 2020,",
    r"course\s+of\s+and\s+\.": "course of 2019 and 2020.",
    r"March\s+\.": "March 2019.",
}

def _fix_ocr_splits(text: str) -> str:
    for pattern, replacement in _OCR_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def remove_inline_footnote_markers(text: str) -> str:
    text = re.sub(r'(?<=[a-zA-Z\)])\d{1,2}(?=[\s.,;:])', "", text)
    return re.sub(r" +", " ", text).strip()


def remove_footnote_blocks(text: str) -> str:
    text = re.sub(
        r'(?<=[a-zA-Z\.\)\'])\d{1,2}(?=\s)',
        ' ',
        text
    )

    lines = []
    for line in text.splitlines():
        stripped = line.strip()

        if re.fullmatch(r"\d{1,2}", stripped):
            continue

        lines.append(line)

    return "\n".join(lines)

def merge_pdf_lines(text: str) -> str:
    lines = text.splitlines()
    merged: list[str] = []
    current = ""
    in_bullet = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if current:
                merged.append(current)
                current = ""
                in_bullet = False
            merged.append("")
            continue

        if stripped.isupper() and len(stripped.split()) < 15:
            if current:
                merged.append(current)
            merged.append(stripped)
            current = ""
            in_bullet = False
            continue

        if stripped.startswith("•"):
            if current:
                merged.append(current)
            current = stripped
            in_bullet = True
            continue

        if in_bullet:
            current += " " + stripped
            continue

        if not current:
            current = stripped
        elif not re.search(r"[.!?:]$", current):
            if looks_like_subheading(stripped):
                merged.append(current)
                merged.append(stripped)
                current = ""
            else:
                current += " " + stripped
        else:
            merged.append(current)
            current = stripped

    if current:
        merged.append(current)

    return "\n".join(merged)

def is_section_heading(line: str) -> bool:
    line = line.strip()
    if not line or line.startswith("•"):
        return False
    words = line.split()
    return line.isupper() and 1 <= len(words) <= 10 and len(line) < 80

def is_bullet_point(line: str) -> bool:
    return line.strip().startswith("•")

def looks_like_subheading(line: str) -> bool:
    line = line.strip()
    if not line or line.startswith("•") or line.isupper() or line.endswith("."):
        return False
    return len(line.split()) <= 12 and line[0].isupper()

def extract_structure(text: str, document_title: str) -> list[dict]:
    lines = text.split("\n")
    records: list[dict] = []

    current_section: str | None = None
    current_subsection: str | None = None
    buffer: list[str] = []

    def flush():
        nonlocal buffer
        content = " ".join(buffer).strip()
        content = re.sub(r"\s+", " ", content)
        content = remove_inline_footnote_markers(content)
        if content:
            records.append({
                "title": document_title,
                "section": current_section,
                "subsection": current_subsection,
                "content": content,
            })
        buffer = []

    stop_patterns = re.compile(
        r"COLLABORATORS:|This project is co-funded by the European Union",
        re.IGNORECASE,
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if stop_patterns.search(line):
            flush()
            break

        if is_section_heading(line):
            flush()
            current_section = line
            current_subsection = None
            continue

        if is_bullet_point(line):
            flush()
            body = line.lstrip("•").strip()

            first_sentence = re.split(r'(?<=[.!?])\s+', body, maxsplit=1)
            raw_label = first_sentence[0]
            if len(raw_label) > 80:
                truncated = raw_label[:80]
                last_space = truncated.rfind(' ')
                raw_label = truncated[:last_space] if last_space != -1 else truncated
            current_subsection = raw_label.rstrip()

            overflow = ""
            if len(first_sentence) > 1:
                overflow = first_sentence[1]
            elif len(body) > len(raw_label):
                overflow = body[len(raw_label):].strip()

            if overflow:
                buffer.append(overflow)
                flush()

            continue

        if looks_like_subheading(line):
            flush()
            current_subsection = line
            continue

        if re.fullmatch(r"\d+", line):
            continue

        buffer.append(line)

    flush()

    merged_records: list[dict] = []
    for rec in records:
        if (
            merged_records
            and len(rec["content"]) < 180
            and rec["section"] == merged_records[-1]["section"]
        ):
            merged_records[-1]["content"] += " " + rec["content"]
        else:
            merged_records.append(rec)

    return merged_records

def chunk_records(
    records: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    )

    chunks: list[dict] = []
    chunk_id_counter = 1
    for rec in records:
        parts = splitter.split_text(rec["content"])
        for i, part in enumerate(parts):
            cleaned = re.sub(r'^[\s.,;:]+', '', part).strip()
            if not cleaned:
                continue
            chunks.append({
                "chunk_id": f"chunk_{chunk_id_counter}",
                "title": rec["title"],
                "section": rec["section"],
                "subsection": rec["subsection"],
                "content": cleaned,
                "char_count": len(cleaned),
                "chunk_index": i,
                "total_chunks_in_record": len(parts),
            })
            chunk_id_counter += 1

    return chunks

def preprocess_and_chunk(
    filepath: str,
    document_title: str,
    extra_noise: set[str] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    text = open(filepath, "r", encoding="utf-8").read()

    text = re.sub(
        rf"^.*?{re.escape(document_title)}",
        document_title,
        text,
        flags=re.S,
    )

    text = clean_pdf_text(text, extra_noise=extra_noise)
    text = remove_footnote_blocks(text)
    text = merge_pdf_lines(text)

    records = extract_structure(text, document_title)
    chunks = chunk_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return chunks

if __name__ == "__main__":
    import json

    DOCUMENT_TITLE = "RESPONSIBLE DATA SHARING WITH DONORS"

    EXTRA_NOISE = {
        "THE CENTRE FOR HUMANITARIAN DATA",
        "DECEMBER 2020",
    }

    chunks = preprocess_and_chunk(
        filepath="output.txt",
        document_title=DOCUMENT_TITLE,
        extra_noise=EXTRA_NOISE,
        chunk_size=500,
        chunk_overlap=50,
    )

    with open("data.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(json.dumps(chunks, indent=2, ensure_ascii=False))
    print(f"\n✓ Total chunks: {len(chunks)}")
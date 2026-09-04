"""What a book and a passage are, everywhere in retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BookRef:
    doc_id: str              # "1_734"
    paper_id: int            # 734
    slot_no: int             # 1
    file_path: str           # "1_734.pdf"
    book_name: str
    book_type: str


@dataclass
class Passage:
    book_name: str
    page_no: int                  # first page of the stretch
    content: str
    page_end: int | None = None   # last page, when it spans more than one




def parse_doc_id(file_path: str) -> tuple[str, int, int] | None:
    """'1_734.pdf' -> ('1_734', 734, 1). Returns None if it isn't that shape."""
    stem = Path(file_path).stem
    match = re.fullmatch(r"(\d+)_(\d+)", stem)
    if not match:
        return None
    slot_no, paper_id = int(match.group(1)), int(match.group(2))
    return stem, paper_id, slot_no

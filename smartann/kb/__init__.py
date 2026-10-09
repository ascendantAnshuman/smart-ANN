"""A user-grown knowledge base, searched beside the fixed SQuAD benchmark index."""

from smartann.kb.sources import fetch_wikipedia, paragraphs_of, read_text_file
from smartann.kb.store import KbHit, KnowledgeBase, SourceInfo, chunk_paragraphs

__all__ = [
    "KbHit",
    "KnowledgeBase",
    "SourceInfo",
    "chunk_paragraphs",
    "fetch_wikipedia",
    "paragraphs_of",
    "read_text_file",
]

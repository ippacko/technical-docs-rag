from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


class Retriever:
    def __init__(self, docs_dir: Path):
        self.chunks = self._load_chunks(docs_dir)
        if not self.chunks:
            raise ValueError(f"No Markdown documents found in {docs_dir}")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)

    @staticmethod
    def _load_chunks(docs_dir: Path) -> list[Chunk]:
        chunks = []
        for path in sorted(docs_dir.glob("*.md")):
            for block in path.read_text(encoding="utf-8").split("\n\n"):
                text = " ".join(line.strip() for line in block.splitlines()).strip()
                if text and not text.startswith("#"):
                    chunks.append(Chunk(source=path.name, text=text))
        return chunks

    def search(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not question.strip():
            raise ValueError("question must not be empty")
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        indexes = scores.argsort()[::-1][: max(1, min(top_k, len(self.chunks)))]
        return [(self.chunks[index], float(scores[index])) for index in indexes]

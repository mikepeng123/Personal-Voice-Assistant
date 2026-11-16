"""Stage 4: long-term memory, searchable by meaning.

Facts are short sentences ("The user prefers morning meetings.") stored in Qdrant with an
embedding from the local embedding model. A new fact that is almost the same as an old one
replaces it, so "I now prefer afternoon meetings" updates the old preference instead of
contradicting it. Appointments never go in here; they live in SQLite.
"""
import logging
import uuid
from datetime import datetime

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

log = logging.getLogger("memory")


class MemoryStore:
    def __init__(self, cfg: dict, collection: str | None = None, qdrant=None, client=None):
        m, mem = cfg["models"], cfg["memory"]
        self.ollama = client or ollama.Client(host=m["ollama_host"], timeout=m.get("request_timeout", 90))
        self.embed_model = m["embed_model"]
        self.keep_alive = m["keep_alive"]
        self.q = qdrant or QdrantClient(url=mem["qdrant_url"])
        self.collection = collection or mem["collection"]
        self.limit = mem["search_limit"]
        self.min_score = mem["min_score"]
        self.same_score = mem["same_score"]
        self.query_prefix = mem.get("query_prefix", "")
        self.doc_prefix = mem.get("document_prefix", "")
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.q.collection_exists(self.collection):
            dims = len(self._embed(self.doc_prefix + "test"))
            self.q.create_collection(self.collection,
                                     vectors_config=VectorParams(size=dims, distance=Distance.COSINE))

    def _embed(self, text: str) -> list[float]:
        return self.ollama.embed(model=self.embed_model, input=text,
                                 keep_alive=self.keep_alive).embeddings[0]

    def search(self, query: str, limit: int | None = None, min_score: float | None = None) -> list[dict]:
        """The memories closest in meaning to the query, best first."""
        vec = self._embed(self.query_prefix + query)
        floor = self.min_score if min_score is None else min_score
        points = self.q.query_points(self.collection, query=vec, limit=limit or self.limit).points
        return [{"id": str(p.id), "text": p.payload["text"], "score": round(p.score, 3)}
                for p in points if p.score >= floor]

    def add(self, text: str, source: str = "tool") -> dict:
        """Store a fact. Returns {"action": "added" | "updated" | "duplicate", ...}."""
        text = " ".join(text.split())
        vec = self._embed(self.doc_prefix + text)
        payload = {"text": text, "source": source, "time": datetime.now().isoformat(timespec="seconds")}
        nearest = self.q.query_points(self.collection, query=vec, limit=1).points
        if nearest and nearest[0].score >= self.same_score:
            old = nearest[0]
            if old.payload["text"].lower() == text.lower():
                return {"action": "duplicate", "id": str(old.id), "text": text}
            self.q.upsert(self.collection, points=[PointStruct(id=old.id, vector=vec, payload=payload)])
            log.info("memory updated: %r -> %r", old.payload["text"], text)
            return {"action": "updated", "id": str(old.id), "text": text, "old": old.payload["text"]}
        new_id = str(uuid.uuid4())
        self.q.upsert(self.collection, points=[PointStruct(id=new_id, vector=vec, payload=payload)])
        log.info("memory added (%s): %r", source, text)
        return {"action": "added", "id": new_id, "text": text}

    def all(self) -> list[dict]:
        points, _ = self.q.scroll(self.collection, limit=1000, with_payload=True, with_vectors=False)
        items = [{"id": str(p.id), **p.payload} for p in points]
        return sorted(items, key=lambda x: x.get("time", ""))

    def delete(self, memory_id: str) -> None:
        self.q.delete(self.collection, points_selector=[memory_id])

    def clear(self) -> None:
        self.q.delete_collection(self.collection)
        self._ensure_collection()

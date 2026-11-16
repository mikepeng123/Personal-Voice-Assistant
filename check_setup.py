"""Stage 0 check: models answer, the chat model can call tools, and both stores write and read."""
import sys
import tempfile

import ollama

import config
from appointments import AppointmentStore

cfg = config.load()
m = cfg["models"]
client = ollama.Client(host=m["ollama_host"])


def chat():
    r = client.chat(model=m["chat_model"], keep_alive=m["keep_alive"],
                    messages=[{"role": "user", "content": "Reply with the single word: ready"}])
    return repr(r.message.content.strip()[:60])


def tool_call():
    spec = [{"type": "function", "function": {
        "name": "get_time", "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}, "required": []}}}]
    r = client.chat(model=m["chat_model"], tools=spec, keep_alive=m["keep_alive"],
                    messages=[{"role": "user", "content": "What time is it? Use the tool."}])
    if not r.message.tool_calls:
        raise RuntimeError("the model did not call the tool; choose a model with tool support")
    return "model called get_time"


def embed():
    r = client.embed(model=m["embed_model"], input="hello world")
    return f"{len(r.embeddings[0])} dimensions"


def sqlite():
    with tempfile.TemporaryDirectory() as d:
        store = AppointmentStore(f"{d}/test.db")
        a = store.create("2030-01-01 09:00", "Test place", "test")
        assert store.get(a["id"])["title"] == "test"
    return "write and read ok"


def qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    vec = client.embed(model=m["embed_model"], input="I like green tea").embeddings[0]
    q = QdrantClient(url=cfg["memory"]["qdrant_url"])
    name = "setup_check"
    if q.collection_exists(name):
        q.delete_collection(name)
    q.create_collection(name, vectors_config=VectorParams(size=len(vec), distance=Distance.COSINE))
    q.upsert(name, points=[PointStruct(id=1, vector=vec, payload={"text": "I like green tea"})])
    hit = q.query_points(name, query=vec, limit=1).points[0]
    q.delete_collection(name)
    return f"read back {hit.payload['text']!r}"


if __name__ == "__main__":
    ok = True
    for label, fn in [("chat model", chat), ("tool calling", tool_call), ("embeddings", embed),
                      ("sqlite", sqlite), ("qdrant", qdrant)]:
        try:
            print(f"PASS  {label}: {fn()}")
        except Exception as e:
            ok = False
            print(f"FAIL  {label}: {e}")
    sys.exit(0 if ok else 1)

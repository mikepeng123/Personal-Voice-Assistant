"""Stage 4 check: remember -> restart -> recall.

The two phases run in separate Python processes, so the second one really is a restart.
Uses its own Qdrant collection and a throwaway calendar, so your real data is untouched.
"""
import subprocess
import sys
import tempfile

import config
from agent import Agent
from appointments import AppointmentStore
from memory import MemoryStore

COLLECTION = "loop_memories_check"


def make_agent(cfg):
    memory = MemoryStore(cfg, collection=COLLECTION)
    agent = Agent(cfg, AppointmentStore(f"{tempfile.mkdtemp()}/check.db"), memory=memory)
    agent.warm_up()
    return agent, memory


def show_memories(memory):
    items = memory.all()
    print("  memories now: " + (" | ".join(f"{m['text']} ({m['source']})" for m in items) or "(none)"))
    return items


def say(agent, text):
    reply = agent.ask(text, on_tool=lambda n, a, r: print(f"  [tool] {n}({a}) -> {r}"))
    if agent.last_memories:
        print(f"  [memory used] {[m['text'] for m in agent.last_memories]}")
    print(f"you> {text}\nbot> {reply}")
    agent.learn_later(text)
    return reply


def phase_remember():
    cfg = config.load()
    agent, memory = make_agent(cfg)
    say(agent, "remember that I prefer morning meetings")
    say(agent, "book a dentist appointment tomorrow at 3 pm in Chicago")
    say(agent, "by the way, my sister Anna lives in Boston")
    print("  (waiting for background extraction ...)")
    agent.wait_for_learning()
    show_memories(memory)


def phase_recall():
    cfg = config.load()
    agent, memory = make_agent(cfg)
    items = show_memories(memory)
    question = "what time of day do I prefer for meetings"
    scores = memory.search(question, limit=5, min_score=0)
    print(f"  similarity to the question (min_score = {memory.min_score}): "
          + ", ".join(f"{m['score']} {m['text']!r}" for m in scores))
    reply = say(agent, question)
    ok = "morning" in reply.lower()
    print(f"\n{'PASS' if ok else 'FAIL'}  the answer says morning")
    texts = " ".join(m["text"].lower() for m in items)
    print(f"{'ok  ' if 'dentist' not in texts else 'WARN'}  no appointment in memory")
    print(f"{'ok  ' if 'anna' in texts else 'info'}  background extraction stored the fact about Anna")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        {"remember": phase_remember, "recall": phase_recall}[sys.argv[1]]()
    else:
        MemoryStore(config.load(), collection=COLLECTION).clear()
        print("=== Phase 1: tell it things ===")
        subprocess.run([sys.executable, __file__, "remember"], check=True)
        print("\n=== Restarting (new process) ===\n=== Phase 2: ask ===")
        sys.exit(subprocess.run([sys.executable, __file__, "recall"]).returncode)

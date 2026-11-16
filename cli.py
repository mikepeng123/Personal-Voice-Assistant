"""Talk to the agent by typing. Shows every tool call so you can see what really happened.

    python cli.py              # replies are spoken
    python cli.py --no-voice   # text only
"""
import argparse
import time

import config
from agent import Agent
from appointments import AppointmentStore
from speech_out import Speaker, ask_and_speak


def show(store):
    appts = store.list_all()
    if not appts:
        print("  (no appointments)")
    for a in appts:
        print(f"  #{a['id']}  {a['date']}  {a['title']}  @ {a['location']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-voice", action="store_true")
    voice = not parser.parse_args().no_voice

    cfg = config.load()
    config.setup_logging(cfg)
    store = AppointmentStore(cfg["paths"]["appointments_db"])
    from memory import MemoryStore
    agent = Agent(cfg, store, memory=MemoryStore(cfg))
    speaker = Speaker(cfg["speech"]["tts_command"]) if voice else None
    first_audio = []
    if speaker:
        speaker.on_start = lambda text: first_audio.append(time.perf_counter())
    print(f"Loading {agent.model} ...")
    agent.warm_up()
    print(f"{agent.name} ready. Commands: /list  /memory  /stop  /reset  /quit")

    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/stop":
            if speaker:
                speaker.stop()
            continue
        if text == "/reset":
            agent.reset()
            print("  (conversation cleared)")
            continue
        if text == "/memory":
            for m in agent.memory.all():
                print(f"  - {m['text']}   ({m.get('source')}, {m.get('time')})")
            continue
        if text == "/list":
            show(store)
            continue

        def on_tool(name, args, result):
            print(f"  [tool] {name}({args})\n         -> {result}")

        first_audio.clear()
        t0 = time.perf_counter()
        if speaker:
            reply = ask_and_speak(agent, speaker, text, on_tool=on_tool)
        else:
            reply = agent.ask(text, on_tool=on_tool)
        timing = f"{time.perf_counter() - t0:.1f}s"
        if first_audio:
            timing = f"first audio {first_audio[0] - t0:.1f}s, done {timing}"
        if agent.last_memories:
            print(f"  [memory] {[m['text'] for m in agent.last_memories]}")
        print(f"{agent.name.lower()}> {reply}   ({timing})")
        agent.learn_later(text, wait=speaker.wait if speaker else None)


if __name__ == "__main__":
    main()

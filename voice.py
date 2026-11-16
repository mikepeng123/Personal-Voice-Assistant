"""Stage 3: the always-on voice assistant in the terminal.

    python voice.py            # talk to it; you can also type commands
    python voice.py --levels   # microphone level meter, for tuning [audio] in config.toml
"""
import argparse
import time

import config
from agent import Agent
from appointments import AppointmentStore
from speech_out import Speaker


def printer(name):
    def on_event(kind, data):
        if kind == "ready":
            print(f"\nListening. Say \"{name}, ...\"   (speech threshold {data:.4f})")
        elif kind == "heard":
            print(f"\n  heard: {data!r}")
        elif kind == "ignored":
            print("  (ignored: no wake word)")
        elif kind == "joined":
            print(f"  joined: {data!r}")
        elif kind == "wake":
            print("  (wake word only: go ahead, no need to say it again)")
        elif kind == "command":
            print(f"  -> {data}")
        elif kind == "tool":
            n, a, r = data
            print(f"  [tool] {n}({a}) -> {r}")
        elif kind == "memories":
            print(f"  [memory] {data}")
        elif kind == "reply":
            print(f"{name.lower()}> {data}")
        elif kind == "turn":
            print(f"  ({data.get('seconds', '?')}s)  [LISTENING - no wake word needed for a few seconds]")
        elif kind == "error":
            print(f"  ERROR: {data}")
    return on_event


def show_levels(cfg):
    from audio_in import Microphone, Segmenter
    a = cfg["audio"]
    mic = Microphone(a["sample_rate"], a["frame_ms"])
    seg = Segmenter(a["sample_rate"], a["frame_ms"], a["start_factor"], a["min_rms"],
                    a["end_silence_s"], a["preroll_s"], a["min_speech_s"], a["max_utterance_s"])
    mic.start()
    print("Stay quiet for a second, then talk. Ctrl+C to quit.")
    n = 0
    while True:
        frame, t = mic.frames.get()
        done = seg.feed(frame, t)
        n += 1
        if done:
            print(f"\n  utterance: {len(done[0]) / a['sample_rate']:.1f}s")
        if n % 3 == 0 and seg.calibrated:
            bar = "#" * min(50, int(seg.last_rms / max(seg.threshold, 1e-6) * 10))
            state = "SPEECH" if seg.active else "      "
            print(f"\r{bar:<50} rms {seg.last_rms:.4f} noise {seg.noise:.4f} "
                  f"threshold {seg.threshold:.4f} {state}", end="", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", action="store_true")
    args = p.parse_args()
    cfg = config.load()
    config.setup_logging(cfg)
    if args.levels:
        return show_levels(cfg)

    from assistant import VoiceAssistant
    from audio_in import Transcriber

    print(f"Loading {cfg['models']['chat_model']} and {cfg['stt']['model']} ...")
    store = AppointmentStore(cfg["paths"]["appointments_db"])
    from memory import MemoryStore
    agent = Agent(cfg, store, memory=MemoryStore(cfg))
    speaker = Speaker(cfg["speech"]["tts_command"])
    va = VoiceAssistant(cfg, agent, speaker, Transcriber(cfg["stt"]),
                        on_event=printer(cfg["assistant"]["name"]))
    va.start()
    print("Type a command, or /mute /unmute /stop /list /memory /quit")
    while True:
        try:
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "/quit":
            break
        elif line == "/mute":
            va.set_muted(True); print("  (muted)")
        elif line == "/unmute":
            va.set_muted(False); print("  (listening)")
        elif line == "/stop":
            va.stop_speaking()
        elif line == "/memory":
            for m in agent.memory.all():
                print(f"  - {m['text']}   ({m.get('source')}, {m.get('time')})")
        elif line == "/list":
            for x in store.list_all():
                print(f"  #{x['id']}  {x['date']}  {x['title']}  @ {x['location']}")
        elif line:
            va.submit_text(line)


if __name__ == "__main__":
    main()

"""Stage 3 check: speak the eight Stage 1 commands. At least five checks must pass.

Uses a fresh database. It tells you what to say next and grades each turn automatically.
Wait for "[LISTENING]" before you speak. Type /skip to give up on a step, /quit or Ctrl+C to stop. Failed transcripts are printed for the README.
"""
import tempfile
import threading
from datetime import date

import config
from agent import Agent
from appointments import AppointmentStore
from assistant import VoiceAssistant
from audio_in import Transcriber
from check_stage1 import make_steps
from speech_out import Speaker
from voice import printer

cfg = config.load()
config.setup_logging(cfg)
name = cfg["assistant"]["name"]
store = AppointmentStore(f"{tempfile.mkdtemp()}/check.db")
steps = make_steps(date.today())
state = {"i": 0, "passed": 0, "failures": []}
done = threading.Event()
show = printer(name)


def prompt():
    if state["i"] >= len(steps):
        return done.set()
    command = steps[state["i"]][0]
    if command == "yes":
        print(f"\n>>> Step {state['i'] + 1}: say \"yes\"  (or \"{name}, yes\" if the follow-up time ran out)")
    else:
        print(f"\n>>> Step {state['i'] + 1}: say \"{name}, {command}\"")


def grade(heard, reply):
    command, checks = steps[state["i"]]
    appts = store.list_all()
    print("  db: " + ("; ".join(f"#{a['id']} {a['date']} {a['title']} @ {a['location']}" for a in appts) or "(empty)"))
    ok_all = True
    for label, fn in checks:
        ok = bool(fn(reply, appts))
        state["passed"] += ok
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok_all:
        state["failures"].append((command, heard, reply))
    state["i"] += 1
    prompt()


def on_event(kind, data):
    show(kind, data)
    if kind == "ready":
        prompt()
    elif kind == "turn" and state["i"] < len(steps):
        grade(data["heard"], data["reply"])


va = VoiceAssistant(cfg, Agent(cfg, store), Speaker(cfg["speech"]["tts_command"]),
                    Transcriber(cfg["stt"]), on_event=on_event)
print("Loading models ...")
va.start()


def keyboard():
    while not done.is_set():
        line = input().strip()
        if line == "/quit":
            done.set()
        elif line == "/skip" and state["i"] < len(steps):
            state["failures"].append((steps[state["i"]][0], "(skipped)", ""))
            state["i"] += 1
            prompt()


threading.Thread(target=keyboard, daemon=True).start()
try:
    while not done.wait(0.2):
        pass
except KeyboardInterrupt:
    pass
print(f"\nPassed {state['passed']}/8 (need at least 5)")
if state["failures"]:
    print("\nFor the README (Stage 3 failures):\n")
    for command, heard, reply in state["failures"]:
        print(f"- Said: \"{name}, {command}\"  \n  Speech-to-text: \"{heard}\"  \n  Reply: \"{reply}\"")

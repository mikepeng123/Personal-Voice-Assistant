"""Stage 2 check: speech starts in under 4 s, and stop() silences it within 0.5 s."""
import sys
import tempfile
import time

import config
from agent import Agent
from appointments import AppointmentStore
from speech_out import Speaker, ask_and_speak

cfg = config.load()
store = AppointmentStore(f"{tempfile.mkdtemp()}/check.db")   # do not touch your real calendar
agent = Agent(cfg, store)
speaker = Speaker(cfg["speech"]["tts_command"])

print(f"warming up {agent.model} ...")
agent.warm_up()

# 1. time to first audio
started = []
speaker.on_start = lambda text: started.append((time.perf_counter(), text))
t0 = time.perf_counter()
reply = ask_and_speak(agent, speaker, "book a dentist appointment tomorrow at 3 pm in Chicago")
generated = time.perf_counter() - t0
speaker.wait(timeout=30)
first = started[0][0] - t0 if started else float("inf")
print(f"reply: {reply}")
print(f"sentences spoken: {len(started)}; full reply generated after {generated:.2f}s")
ok1 = first < 4
print(f"{'PASS' if ok1 else 'FAIL'}  first audio after {first:.2f}s (need < 4)")

# 2. stop mid-sentence
started.clear()
speaker.say("This is a long test sentence that keeps going and going so that there is "
            "plenty of time to interrupt it in the middle before it reaches the end.")
speaker.say("This second sentence should never be heard.")
while not started:
    time.sleep(0.01)
time.sleep(1.0)
t1 = time.perf_counter()
speaker.stop()
speaker.wait(timeout=5)
silence = time.perf_counter() - t1
ok2 = silence < 0.5 and len(started) == 1
print(f"{'PASS' if ok2 else 'FAIL'}  silent {silence:.3f}s after stop (need < 0.5), "
      f"queued sentence dropped: {len(started) == 1}")
sys.exit(0 if ok1 and ok2 else 1)

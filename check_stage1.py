"""Stage 1 check: runs the eight checks against a fresh database and prints a README table row.

    python check_stage1.py                         # model from config.toml
    python check_stage1.py --models gemma4:12b-mlx qwen3:8b
"""
import argparse
import re
import tempfile
import time
from datetime import date, timedelta

import config
from agent import Agent
from appointments import AppointmentStore


def fridays(today: date) -> set[str]:
    ahead = (4 - today.weekday()) % 7 or 7
    out = {(today + timedelta(days=ahead)).isoformat()}
    if today.weekday() == 4:
        out.add(today.isoformat())  # "on Friday" said on a Friday is ambiguous; accept both
    return out


def make_steps(today: date):
    """The eight checks as (command, [(label, check(reply, appointments))]). Also used by Stage 3."""
    tomorrow = (today + timedelta(days=1)).isoformat()
    fri = fridays(today)
    s = {}  # shared state between checks

    def check1(r, a):
        if len(a) == 1:
            s["id"] = a[0]["id"]
        return len(a) == 1 and a[0]["date"] == f"{tomorrow} 15:00" and "chicago" in a[0]["location"].lower()

    return [
        ("book a dentist appointment tomorrow at 3 pm in Chicago", [
            ("1 one appointment, tomorrow 15:00, Chicago", check1),
            ("2 the id is in the answer", lambda r, a: "id" in s and re.search(rf"\b{s['id']}\b", r) is not None),
        ]),
        ("book a haircut on Friday at 10 am", [
            ("3 nothing saved, asks where", lambda r, a: len(a) == 1 and "?" in r),
        ]),
        ("what is on my calendar", [
            ("4 dentist mentioned", lambda r, a: "dentist" in r.lower()),
        ]),
        ("move my dentist appointment to 4 pm", [
            ("5 same id, 16:00, no duplicate", lambda r, a: len(a) == 1 and a[0]["id"] == s.get("id")
             and a[0]["date"] == f"{tomorrow} 16:00"),
        ]),
        ("delete my dentist appointment", [
            ("6 asks first, still there", lambda r, a: len(a) == 1 and "?" in r),
        ]),
        ("yes", [
            ("7 deleted", lambda r, a: len(a) == 0),
        ]),
        ("book lunch with Sam on Friday at noon at Cafe Roma", [
            ("8 exactly one appointment", lambda r, a: len(a) == 1 and a[0]["date"][:10] in fri
             and a[0]["date"].endswith("12:00") and "roma" in a[0]["location"].lower()),
        ]),
    ]


def run(cfg, model):
    store = AppointmentStore(f"{tempfile.mkdtemp()}/check.db")
    agent = Agent(cfg, store, model=model)
    steps = make_steps(date.today())

    print(f"\n=== {agent.model} ===  (warming up)")
    agent.warm_up()
    passed, total = 0, 0.0
    for command, checks in steps:
        t0 = time.perf_counter()
        reply = agent.ask(command)
        total += time.perf_counter() - t0
        appts = store.list_all()
        print(f"\nyou> {command}\nbot> {reply}")
        print("db:  " + ("; ".join(f"#{a['id']} {a['date']} {a['title']} @ {a['location']}" for a in appts) or "(empty)"))
        for label, fn in checks:
            ok = bool(fn(reply, appts))
            passed += ok
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"\n{agent.model}: {passed}/8 passed in {total:.1f}s")
    return agent.model, passed, total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", help="Ollama model names (default: chat_model in config.toml)")
    args = p.parse_args()
    cfg = config.load()
    rows = []
    for model in args.models or [cfg["models"]["chat_model"]]:
        try:
            rows.append(run(cfg, model))
        except Exception as e:
            print(f"\n{model}: could not run ({e}). Is it pulled? Try: ollama pull {model}")
    print("\n| Model | Checks passed (of 8) | Total seconds for the eight checks |")
    print("|---|---|---|")
    for model, passed, secs in rows:
        print(f"| {model} | {passed} | {secs:.1f} |")

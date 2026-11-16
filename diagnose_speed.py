"""Where do the seconds go? Runs the booking request step by step and prints Ollama's own timings.

    python diagnose_speed.py
"""
import tempfile
import time

import config
from agent import Agent, build_system_prompt
from appointments import AppointmentStore

REQUEST = "book a dentist appointment tomorrow at 3 pm in Chicago"
cfg = config.load()
agent = Agent(cfg, AppointmentStore(f"{tempfile.mkdtemp()}/diag.db"))
client = agent.client
sec = lambda ns: (ns or 0) / 1e9


def show_loaded():
    for m in client.ps().models:
        if m.model.startswith(agent.model) or m.name.startswith(agent.model):
            gpu = 100 * (m.size_vram or 0) / max(m.size or 1, 1)
            print(f"  loaded: {m.name}  size {m.size / 1e9:.1f} GB  on GPU {gpu:.0f}%  "
                  f"context {getattr(m, 'context_length', '?')}")
            return
    print("  model is NOT loaded right now (the next request has to load it)")


def run(label, think):
    print(f"\n=== {label} ===")
    extra = {} if think is None else {"think": think}
    messages = [{"role": "system", "content": build_system_prompt(agent.name)},
                {"role": "user", "content": f"[{time.strftime('%H:%M')}] {REQUEST}"}]
    t_request = time.perf_counter()
    for step in (1, 2, 3):
        t0, first, thinking, content, calls, final = time.perf_counter(), None, "", "", [], None
        try:
            stream = client.chat(model=agent.model, messages=messages, tools=agent.tool_specs,
                                 options=agent.options, keep_alive=agent.keep_alive,
                                 stream=True, **extra)
            for chunk in stream:
                m = chunk.message
                if first is None and (m.content or m.tool_calls or getattr(m, "thinking", None)):
                    first = time.perf_counter() - t0
                thinking += getattr(m, "thinking", None) or ""
                content += m.content or ""
                calls += m.tool_calls or []
                if chunk.done:
                    final = chunk
        except Exception as e:
            print(f"  error: {e}")
            return
        total = time.perf_counter() - t0
        print(f"step {step}: {total:.2f}s total, first output after {first or 0:.2f}s")
        if final:
            print(f"  load {sec(final.load_duration):.2f}s | "
                  f"prompt {final.prompt_eval_count or 0} tokens in {sec(final.prompt_eval_duration):.2f}s | "
                  f"generated {final.eval_count or 0} tokens in {sec(final.eval_duration):.2f}s")
        print(f"  hidden thinking: {len(thinking)} characters" + (f" -> {thinking[:150]!r}..." if thinking else ""))
        if calls:
            print(f"  tool calls: {[c.function.name for c in calls]}")
        if content.strip():
            print(f"  text: {content.strip()[:150]!r}")
        if not calls:
            break
        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {"function": {"name": c.function.name, "arguments": dict(c.function.arguments or {})}} for c in calls]})
        for c in calls:
            result = agent.tools.run(c.function.name, dict(c.function.arguments or {}), [REQUEST])
            messages.append({"role": "tool", "content": result, "tool_name": c.function.name})
    print(f"whole request: {time.perf_counter() - t_request:.2f}s")


print(f"model: {agent.model}")
show_loaded()
print("warming up (system prompt + tools) ...")
agent.warm_up()
show_loaded()
think = agent.extra.get("think")
run(f"first request after warm-up (think={think})", think=think)
run(f"same request again (think={think})", think=think)

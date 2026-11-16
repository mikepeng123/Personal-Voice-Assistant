"""A small tool-calling agent on top of any Ollama chat model named in config.toml."""
import json
import logging
import queue
import re
import threading
from datetime import datetime, timedelta

import httpx
import ollama

from appointments import AppointmentStore
from tools import ToolRunner, confirmation, looks_like_appointment

log = logging.getLogger("agent")

SYSTEM_PROMPT = """You are {name}, a voice assistant that manages the user's appointments.

Today is {today}. Tomorrow is {tomorrow}. The next seven days are: {days}.
Each user message starts with the current time in brackets, like [14:05].

Rules:
- Use the tools for every booking, change, lookup and deletion. Dates are "YYYY-MM-DD HH:MM" on a 24-hour clock.
- If the user has not said where, ask one short question about the location. Never invent a location.
- Never say something was booked, moved or deleted unless the tool just returned ok true. When you book something, say the id from the tool result.
- To move or change an appointment, find its id with list_appointments and use update_appointment. Never create a second one.
- To delete: find it with list_appointments, ask the user to confirm with yes or no, and call delete_appointment only after they say yes.
- If a tool returns an error, follow what the error says.
- When you call a tool, write no other text in that message. Speak only after you have the result.
- Reply in one or two short spoken sentences. Keep the first sentence very short, like "Booked, the id is 4."
- Say dates and times the way people speak, like "tomorrow at 3 PM", never like "2026-11-16 15:00".
- No lists, no markdown, no emoji.
- A user message may end with "(Memory: ...)": facts you remembered about the user earlier. Use them when they help.
- When the user asks you to remember something, call remember. Only personal facts and preferences, never appointments."""

EXTRACT_PROMPT = """(Internal memory task. This reply is not spoken and not shown to the user.)
From the user's last message, list lasting personal facts or preferences about the user that are worth remembering,
for example "The user prefers morning meetings." or "The user's sister Anna lives in Boston."
Do not include appointments, bookings or anything tied to a specific date or time. Do not include requests or questions.
The user's last message was: "{text}"
Reply with only a JSON array of short sentences, or [] if there is nothing worth remembering."""

FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'd|my|me|mine|we|our|us)\b", re.I)
MEMORY_BLOCK = re.compile(r"\n\(Memory: .*\)$", re.S)


def parse_facts(raw: str) -> list[str]:
    match = re.search(r"\[.*\]", raw or "", re.S)
    try:
        facts = json.loads(match.group()) if match else []
    except json.JSONDecodeError:
        return []
    return [f.strip() for f in facts if isinstance(f, str) and 3 < len(f.strip()) < 200]


TIME_TAG = re.compile(r"^\[\d{2}:\d{2}\]\s*")


def build_system_prompt(name: str, now: datetime | None = None) -> str:
    """Only the date goes in here, so the prompt stays the same all day and Ollama can
    reuse its cached processing of it. The time of day goes in each user message."""
    now = now or datetime.now()
    days = ", ".join(
        f"{d:%A} {d:%Y-%m-%d}" for d in (now + timedelta(days=i) for i in range(1, 8))
    )
    return SYSTEM_PROMPT.format(
        name=name,
        today=now.strftime("%A %Y-%m-%d"),
        tomorrow=(now + timedelta(days=1)).strftime("%A %Y-%m-%d"),
        days=days,
    )


class Agent:
    def __init__(self, cfg: dict, store: AppointmentStore, model: str | None = None, client=None,
                 memory=None):
        m, a = cfg["models"], cfg["assistant"]
        self.client = client or ollama.Client(host=m["ollama_host"], timeout=m.get("request_timeout", 90))
        self.model = model or m["chat_model"]
        self.keep_alive = m["keep_alive"]
        self.options = {"temperature": m["temperature"]}
        # Optional: think = false in config.toml turns off hidden reasoning on models that have it.
        # It is only sent to models that support thinking, so any model can be swapped in.
        self.extra = {"think": m["think"]} if "think" in m else {}
        if self.extra and client is None:
            try:
                caps = getattr(self.client.show(self.model), "capabilities", None)
            except Exception:
                caps = None  # unknown (old Ollama or model not pulled): keep the setting
            if caps is not None and "thinking" not in caps:
                self.extra = {}
        self.name = a["name"]
        self.max_turns = a["max_history_turns"]
        self.max_steps = a["max_tool_steps"]
        self.fast_confirmations = a.get("fast_confirmations", False)
        self.memory = memory
        self.tools = ToolRunner(store, memory)
        self.tool_specs = self.tools.specs
        self.history: list[dict] = []
        self.last_memories: list[dict] = []     # memories put into the last request (for the UI)
        self.last_tool_names: list[str] = []
        self._learn_queue = queue.Queue()
        if memory:
            threading.Thread(target=self._learn_worker, daemon=True, name="memory").start()

    def reset(self):
        self.history = []

    def warm_up(self):
        """Load the model and fill Ollama's prompt cache with the system prompt and tools,
        so the first real request does not pay for processing them."""
        self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": build_system_prompt(self.name)},
                      {"role": "user", "content": "[00:00] hi"}],
            tools=self.tool_specs,
            options={**self.options, "num_predict": 1},
            keep_alive=self.keep_alive,
            **self.extra,
        )

    def _trim(self):
        """Keep the last N user turns, always cutting at a user message."""
        starts = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(starts) > self.max_turns:
            self.history = self.history[starts[-self.max_turns]:]

    def ask(self, text: str, on_tool=None, on_text=None) -> str:
        """Run one user request to completion.

        on_tool(name, args, result) is called for each tool call.
        on_text(chunk) is called with reply text as it streams, so speech can start early.
        """
        # Memories relevant to this request go at the end of the user message. The message
        # (time tag and memories included) stays in the history unchanged, so it stays cached.
        content = f"[{datetime.now():%H:%M}] {text}"
        self.last_memories = self._recall(text)
        if self.last_memories:
            content += "\n(Memory: " + " ".join(m["text"] for m in self.last_memories) + ")"
        self.history.append({"role": "user", "content": content})
        self._trim()
        user_texts = [MEMORY_BLOCK.sub("", TIME_TAG.sub("", m["content"]))
                      for m in self.history if m["role"] == "user"]
        self.last_tool_names = []

        for _ in range(self.max_steps):
            messages = [{"role": "system", "content": build_system_prompt(self.name)}] + self.history
            content, calls = "", []
            try:
                stream = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_specs,
                    options=self.options,
                    keep_alive=self.keep_alive,
                    stream=True,
                    **self.extra,
                )
                for chunk in stream:
                    part = chunk.message
                    if part.content:
                        content += part.content
                        if on_text:
                            on_text(part.content)
                    if part.tool_calls:
                        calls.extend(part.tool_calls)
            except (httpx.TimeoutException, httpx.TransportError, ollama.ResponseError) as e:
                log.error("model request failed: %r", e)
                reply = "Sorry, the language model did not answer. Please try again."
                if on_text:
                    on_text(reply)
                return reply
            entry = {"role": "assistant", "content": content}
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.function.name, "arguments": dict(c.function.arguments or {})}}
                    for c in calls
                ]
            self.history.append(entry)

            if not calls:
                reply = content.strip()
                log.info("reply: %s", reply)
                return reply

            results = []
            for c in calls:
                name, args = c.function.name, dict(c.function.arguments or {})
                self.last_tool_names.append(name)
                result = self.tools.run(name, args, user_texts)
                log.info("tool %s(%s) -> %s", name, args, result)
                if on_tool:
                    on_tool(name, args, result)
                self.history.append({"role": "tool", "content": result, "tool_name": name})
                results.append((name, result))

            # Fast path: a successful booking, move or delete is confirmed straight from the
            # tool result, which saves a whole second model call (about 1.5 s).
            reply = confirmation(results) if self.fast_confirmations else None
            if reply:
                if on_text:
                    on_text(reply)
                self.history.append({"role": "assistant", "content": reply})
                log.info("reply (from tool result): %s", reply)
                return reply

        return "Sorry, I got stuck on that. Please try again."

    # ---------- memory ----------

    def _recall(self, text: str) -> list[dict]:
        if not self.memory:
            return []
        try:
            return self.memory.search(text)
        except Exception as e:  # memory is a nice-to-have; never let it break a request
            log.warning("memory search failed: %s", e)
            return []

    def learn_later(self, text: str, wait=None) -> None:
        """After the reply, in the background: extract personal facts from what the user said.
        wait() is called first (e.g. speaker.wait) so this never competes with the reply."""
        if self.memory:
            self._learn_queue.put((text, list(self.history), list(self.last_tool_names), wait))

    def wait_for_learning(self) -> None:
        self._learn_queue.join()

    def _learn_worker(self):
        while True:
            text, history, tools_used, wait = self._learn_queue.get()
            try:
                if wait:
                    wait()
                for fact in self.extract_facts(text, history, tools_used):
                    result = self.memory.add(fact, source="extracted")
                    log.info("learned (%s): %s", result["action"], fact)
            except Exception as e:
                log.warning("memory extraction failed: %s", e)
            finally:
                self._learn_queue.task_done()

    def extract_facts(self, text: str, history: list[dict], tools_used: list[str]) -> list[str]:
        if "remember" in tools_used or not FIRST_PERSON.search(text):
            return []  # already saved by the tool, or nothing about the user
        # The extraction question is added after the real conversation, so the system prompt
        # and history are the same as in the cache and nothing has to be processed again.
        messages = ([{"role": "system", "content": build_system_prompt(self.name)}] + history
                    + [{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}])
        resp = self.client.chat(model=self.model, messages=messages, tools=self.tool_specs,
                                options={**self.options, "temperature": 0, "num_predict": 200},
                                keep_alive=self.keep_alive, **self.extra)
        facts = parse_facts(resp.message.content)
        kept = [f for f in facts if not looks_like_appointment(f)]
        log.info("extracted from %r: %s (dropped as appointments: %s)", text, kept,
                 [f for f in facts if f not in kept])
        return kept

"""Stage 2: speech out.

The reply streams in from the model, is cut into sentences, and each finished sentence is
spoken right away by a background thread. stop() cuts speech off immediately by killing the
text-to-speech process and dropping every sentence still queued for that reply.
"""
import logging
import queue
import re
import subprocess
import threading
import time

log = logging.getLogger("speech")

# ---------- text cleanup ----------

EMOJI = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F]")


def clean_for_speech(text: str) -> str:
    """Remove things a voice reads badly: markdown, links, bullets, emoji."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)          # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)                       # bare urls
    text = re.sub(r"^\s*([-*+•]|\d+[.)])\s+", "", text, flags=re.M)  # list markers
    text = re.sub(r"[*#`_~>|]", "", text)                          # markdown symbols
    text = EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lstrip("-").strip()


# ---------- sentence splitting ----------

ABBREVIATIONS = {"a.m.", "p.m.", "dr.", "mr.", "mrs.", "ms.", "st.", "e.g.", "i.e.", "etc.", "vs."}
BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")


class SentenceSplitter:
    """feed() streamed text in; get back the sentences that are complete so far."""

    def __init__(self):
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        sentences, start = [], 0
        for m in BOUNDARY.finditer(self.buffer):
            candidate = self.buffer[start:m.end()].strip()
            if candidate and candidate.split()[-1].lower() in ABBREVIATIONS:
                continue  # "3 p.m. in Chicago" is one sentence
            if candidate:
                sentences.append(candidate)
            start = m.end()
        self.buffer = self.buffer[start:]
        return sentences

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest else []


# ---------- the speaker ----------

class Speaker:
    """Speaks sentences one after another in a background thread.

    command is the TTS program from config.toml, e.g. ["say", "-r", "190"];
    the sentence is added as the last argument.
    """

    def __init__(self, command: list[str]):
        self.command = list(command)
        self.on_start = None            # optional callback(text) when a sentence starts playing
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._turn = 0                  # bumped by stop(); stale sentences are dropped
        self._pending = 0               # sentences queued or playing
        self._proc = None
        threading.Thread(target=self._worker, daemon=True, name="speaker").start()

    @property
    def speaking(self) -> bool:
        """True while anything is playing or waiting to play (Stage 3 mutes the mic on this)."""
        return self._pending > 0

    def begin(self) -> int:
        """Start a new reply: silence the old one and return the token for the new one."""
        self.stop()
        return self._turn

    def say(self, text: str, turn: int | None = None) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        with self._lock:
            turn = self._turn if turn is None else turn
            if turn != self._turn:
                return  # this reply was stopped
            self._pending += 1
            self._queue.put((turn, text))

    def stop(self) -> None:
        """Silence now and drop everything queued."""
        with self._lock:
            self._turn += 1
            while True:
                try:
                    self._queue.get_nowait()
                    self._pending -= 1
                except queue.Empty:
                    break
            proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()

    def wait(self, timeout: float | None = None) -> None:
        end = None if timeout is None else time.monotonic() + timeout
        while self.speaking and (end is None or time.monotonic() < end):
            time.sleep(0.02)

    def _worker(self):
        while True:
            turn, text = self._queue.get()
            with self._lock:
                if turn != self._turn:
                    self._pending -= 1
                    continue
                try:
                    proc = self._proc = subprocess.Popen(
                        self.command + [text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except OSError as e:  # e.g. TTS program not installed: skip, but keep the thread alive
                    log.error("text-to-speech failed (%s): %s", self.command[0], e)
                    self._pending -= 1
                    continue
            if self.on_start:
                self.on_start(text)
            proc.wait()
            with self._lock:
                self._proc = None
                self._pending -= 1


def ask_and_speak(agent, speaker: Speaker, text: str, on_tool=None) -> str:
    """Send text to the agent and speak each sentence of the reply as soon as it is complete."""
    turn = speaker.begin()
    splitter = SentenceSplitter()

    def on_text(chunk):
        for sentence in splitter.feed(chunk):
            speaker.say(sentence, turn)

    reply = agent.ask(text, on_tool=on_tool, on_text=on_text)
    for sentence in splitter.flush():
        speaker.say(sentence, turn)
    return reply

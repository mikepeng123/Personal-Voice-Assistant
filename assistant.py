"""The always-on loop: microphone -> speech-to-text -> wake word -> agent -> speech.

Runs in background threads started once by start(). Anything else (the terminal now, the web
page in Stage 5) only reads its state and calls submit_text / set_muted / stop_speaking.
"""
import logging
import queue
import subprocess
import threading
import time
from collections import deque

from audio_in import Microphone, Segmenter
from speech_out import ask_and_speak
from wake import FILLERS, find_wake, normalize

log = logging.getLogger("assistant")


class VoiceAssistant:
    def __init__(self, cfg, agent, speaker, transcriber, on_event=None):
        a, w = cfg["audio"], cfg["wake"]
        self.agent, self.speaker, self.transcriber = agent, speaker, transcriber
        self.on_event = on_event or (lambda kind, data: None)
        self.wake_words = {x.lower() for x in w["words"]}
        self.follow_up_s = w["follow_up_s"]
        self.ignore = {normalize(p) for p in cfg["stt"].get("ignore_phrases", [])}
        self.unmute_delay = a["unmute_delay_s"]
        self.tone = a.get("listen_tone") or []
        self.mic = Microphone(a["sample_rate"], a["frame_ms"], self.is_muted)
        self.segmenter = Segmenter(
            a["sample_rate"], a["frame_ms"], a["start_factor"], a["min_rms"], a["end_silence_s"],
            a["preroll_s"], a["min_speech_s"], a["max_utterance_s"])

        self.inbox = queue.Queue()          # ("heard" | "typed", text, start_time, end_time)
        self._held = None                   # an item taken out while joining, handled next
        self.join_gap = a.get("join_gap_s", 0.5)
        self._transcribing_since = None     # start time of the utterance being transcribed
        self.turns = []                     # finished turns, newest last (for the web page)
        self.current = None                 # the turn being handled right now, if any
        self.events = deque(maxlen=30)      # recent speech: heard, ignored, wake, joined, error
        self.status_log = deque(maxlen=12)  # (time, status) each time the status changes
        self.models_ready = False
        self.user_muted = False
        self.busy = False                   # handling a request (thinking or speaking)
        self.unmute_at = 0.0                # mic stays deaf until then (echo tail, tone)
        self.follow_up_until = 0.0          # wake word not needed until then
        self._started = False
        self._lock = threading.Lock()

    # ---------- state the UI reads ----------

    @property
    def status(self) -> str:
        if not (self.models_ready and self.segmenter.calibrated):
            return "loading"
        if self.user_muted:
            return "muted"
        if self.speaker.speaking:
            return "speaking"
        if self.busy:
            return "thinking"
        return "listening"

    @property
    def follow_up_left(self) -> float:
        return max(0.0, self.follow_up_until - time.monotonic())

    def is_muted(self) -> bool:
        return (self.user_muted or self.busy or self.speaker.speaking
                or time.monotonic() < self.unmute_at)

    # ---------- controls ----------

    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._listen, daemon=True, name="listener").start()
        threading.Thread(target=self._work, daemon=True, name="worker").start()
        threading.Thread(target=self._watch_status, daemon=True, name="status").start()

    def submit_text(self, text: str):
        if text.strip():
            now = time.monotonic()
            self.inbox.put(("typed", text.strip(), now, now))

    def set_muted(self, muted: bool):
        self.user_muted = muted
        self._emit("status", self.status)

    def stop_speaking(self):
        self.speaker.stop()

    # ---------- threads ----------

    def _watch_status(self):
        last = None
        while True:
            now = self.status
            if now != last:
                self.status_log.append((time.strftime("%H:%M:%S"), now))
                last = now
            time.sleep(0.05)

    def _emit(self, kind, data=None):
        if kind in ("heard", "ignored", "wake", "joined", "error"):
            self.events.append((time.strftime("%H:%M:%S"), kind, str(data)))
        try:
            self.on_event(kind, data)
        except Exception:
            log.exception("on_event failed")

    def _listen(self):
        self._emit("status", "loading")
        self.agent.warm_up()
        self.transcriber.warm_up()
        self.models_ready = True
        self.mic.start()
        announced = False
        while True:
            item = self.mic.frames.get()
            if item is None:            # muting started: drop any half utterance
                self.segmenter.reset()
                continue
            done = self.segmenter.feed(*item)
            if not announced and self.segmenter.calibrated:
                announced = True
                self._emit("ready", self.segmenter.threshold)
            if done:
                audio, start = done
                end = start + len(audio) / self.mic.sample_rate
                self._transcribing_since = start
                try:
                    text = self.transcriber(audio)
                except Exception as e:
                    log.exception("speech-to-text failed")
                    self._emit("error", f"speech-to-text failed: {e}")
                    continue
                finally:
                    self._transcribing_since = None
                self._emit("heard", text)
                self.inbox.put(("heard", text, start, end))

    def _next(self, timeout=None):
        if self._held:
            item, self._held = self._held, None
            return item
        return self.inbox.get(timeout=timeout)

    def _still_talking(self, end: float) -> bool:
        """Is a new utterance, which began right after `end`, being recorded or transcribed?"""
        seg, t = self.segmenter, self._transcribing_since
        return ((seg.active and seg.start_t - end <= self.join_gap)
                or (t is not None and t - end <= self.join_gap))

    def _gather(self, text: str, end: float) -> str:
        """People pause mid-sentence ("Loop ... book a haircut"). If the next utterance started
        within join_gap_s of this one ending, wait for it and join the pieces into one."""
        give_up = time.monotonic() + 10
        while time.monotonic() < give_up:
            try:
                item = self._next(timeout=0.1)
            except queue.Empty:
                if self._still_talking(end):
                    continue
                break
            kind, more, start, more_end = item
            if kind == "heard" and start - end <= self.join_gap:
                text, end = f"{text} {more}".strip(), more_end
                continue
            self._held = item
            break
        return text

    def _work(self):
        while True:
            kind, text, start, end = self._next()
            if kind == "heard":
                if start < self.unmute_at:
                    continue  # recorded before the last reply ended
                joined = self._gather(text, end)
                if joined != text:
                    self._emit("joined", joined)
                text = joined
            try:
                self._handle(kind, text, start)
            except Exception as e:
                log.exception("turn failed")
                self._emit("error", str(e))
                self.busy = False

    def _handle(self, kind, text, start):
        if kind == "typed":
            return self._turn(heard=text, command=text, typed=True)
        if not text or normalize(text) in self.ignore:
            return self._emit("ignored", text)
        command = find_wake(text, self.wake_words)
        if command is None:
            if start > self.follow_up_until or set(normalize(text).split()) <= FILLERS:
                return self._emit("ignored", text)
            command = text                       # follow-up: no wake word needed
        if not command:
            # Only "Loop": keep the mic open (no tone, no deaf period, the user is about to
            # speak) and accept the next utterance without the wake word.
            self.follow_up_until = time.monotonic() + self.follow_up_s
            return self._emit("wake", text)
        self._turn(heard=text, command=command)

    def _turn(self, heard, command, typed=False):
        self.busy = True
        turn = {"heard": heard, "command": command, "tools": [], "reply": "",
                "time": time.strftime("%H:%M:%S"), "typed": typed}
        self.current = turn
        self._emit("command", command)
        self._emit("status", "thinking")
        t0 = time.perf_counter()

        def on_tool(name, args, result):
            turn["tools"].append({"name": name, "args": args, "result": result})
            self._emit("tool", (name, args, result))

        try:
            turn["reply"] = ask_and_speak(self.agent, self.speaker, command, on_tool=on_tool)
            turn["seconds"] = round(time.perf_counter() - t0, 2)
            turn["memories"] = [m["text"] for m in self.agent.last_memories]
            if turn["memories"]:
                self._emit("memories", turn["memories"])
            self._emit("reply", turn["reply"])
            self.speaker.wait()
            self.agent.learn_later(command)   # background, after speaking
        finally:
            self.turns = (self.turns + [turn])[-50:]
            self.current = None
            self._listen_again()
            self._emit("turn", turn)   # only now is the mic open again

    def _listen_again(self):
        """Tone, short deaf period for the echo, then accept speech without the wake word."""
        self.busy = True
        if self.tone:
            try:
                subprocess.run(self.tone, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
            except (OSError, subprocess.TimeoutExpired) as e:
                log.warning("listen tone failed: %s", e)
        self.unmute_at = time.monotonic() + self.unmute_delay
        self.follow_up_until = self.unmute_at + self.follow_up_s
        time.sleep(self.unmute_delay)  # still deaf: wait it out before saying "listening"
        self.busy = False
        self._emit("status", "listening")

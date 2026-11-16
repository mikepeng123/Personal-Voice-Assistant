"""Stage 3: microphone, utterance segmentation and speech-to-text."""
import logging
import queue
import time
from collections import deque

import numpy as np

log = logging.getLogger("audio_in")


class Microphone:
    """Captures the microphone continuously in small frames.

    Frames recorded while is_muted() is true are thrown away, and a None marker is queued
    when muting starts, so a half-recorded utterance is discarded instead of finished later.
    """

    def __init__(self, sample_rate: int, frame_ms: int, is_muted=lambda: False):
        self.sample_rate = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.is_muted = is_muted
        self.frames = queue.Queue()
        self._was_muted = False
        self._stream = None

    def start(self):
        import sounddevice as sd  # imported here so the rest of the project works without it
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                      blocksize=self.frame, callback=self._callback)
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        if self.is_muted():
            if not self._was_muted:
                self.frames.put(None)
                self._was_muted = True
            return
        self._was_muted = False
        self.frames.put((indata[:, 0].copy(), time.monotonic()))


class Segmenter:
    """Cuts the frame stream into utterances using loudness.

    Speech starts when the volume rises above the room's noise level (measured at start-up and
    slowly updated), and ends after end_silence_s of quiet. A short pre-roll of audio from just
    before the start is kept so the first syllable survives.
    """

    def __init__(self, sample_rate, frame_ms, start_factor, min_rms, end_silence_s,
                 preroll_s, min_speech_s, max_utterance_s, calibrate_s=1.0):
        self.frame_s = frame_ms / 1000
        frames = lambda seconds: max(1, round(seconds / self.frame_s))
        self.end_frames = frames(end_silence_s)
        self.min_voiced = frames(min_speech_s)
        self.max_frames = frames(max_utterance_s)
        self.calibrate_frames = frames(calibrate_s)
        self.preroll = deque(maxlen=frames(preroll_s))
        self.factor, self.min_rms = start_factor, min_rms
        self.noise, self.last_rms, self._seen = 0.0, 0.0, 0
        self.reset()

    @property
    def calibrated(self) -> bool:
        return self._seen >= self.calibrate_frames

    @property
    def threshold(self) -> float:
        return max(self.noise * self.factor, self.min_rms)

    def reset(self):
        self.active, self.frames, self.levels = False, [], []
        self.silent = self.voiced = self.loud = 0
        self.start_t = 0.0
        self.preroll.clear()

    def feed(self, frame: np.ndarray, t: float):
        """Returns (audio, start_time) when an utterance has just ended, else None."""
        rms = self.last_rms = float(np.sqrt(np.mean(frame ** 2)))
        if not self.calibrated:  # first second: measure the room
            self._seen += 1
            self.noise += (rms - self.noise) / self._seen
            return None

        threshold = self.threshold
        if not self.active:
            self.preroll.append(frame)
            if rms > threshold:
                self.loud += 1
                if self.loud >= 2:  # two loud frames in a row: speech started
                    self.active = True
                    self.frames = list(self.preroll)
                    self.start_t = t - (len(self.frames) - 1) * self.frame_s
                    self.voiced, self.silent = self.loud, 0
            else:
                self.loud = 0
                self.noise = 0.98 * self.noise + 0.02 * rms  # follow slow changes in the room
            return None

        self.frames.append(frame)
        self.levels.append(rms)
        if rms > threshold * 0.6:
            self.voiced += 1
            self.silent = 0
        else:
            self.silent += 1

        too_long = len(self.frames) >= self.max_frames
        if self.silent >= self.end_frames or too_long:
            if too_long:  # never went quiet: the room probably got louder
                self.noise = max(self.noise, float(np.median(self.levels)) / self.factor * 1.2)
            audio, start, voiced = np.concatenate(self.frames), self.start_t, self.voiced
            self.reset()
            if voiced >= self.min_voiced:
                return audio, start
        return None


class Transcriber:
    """Local speech-to-text, engine chosen in config.toml."""

    def __init__(self, stt: dict):
        self.engine, self.model = stt["engine"], stt["model"]
        self.language = stt.get("language") or None
        self.prompt = stt.get("initial_prompt") or None
        if self.engine == "mlx-whisper":
            import mlx_whisper
            self._mlx = mlx_whisper
        elif self.engine == "faster-whisper":
            from faster_whisper import WhisperModel
            self._fw = WhisperModel(self.model, device="auto", compute_type="int8")
        else:
            raise ValueError(f"unknown stt engine {self.engine!r}")

    def __call__(self, audio: np.ndarray) -> str:
        audio = audio.astype(np.float32)
        t0 = time.perf_counter()
        if self.engine == "mlx-whisper":
            text = self._mlx.transcribe(audio, path_or_hf_repo=self.model, language=self.language,
                                        initial_prompt=self.prompt,
                                        condition_on_previous_text=False)["text"]
        else:
            segments, _ = self._fw.transcribe(audio, language=self.language, beam_size=1,
                                              initial_prompt=self.prompt,
                                              condition_on_previous_text=False)
            text = "".join(s.text for s in segments)
        log.info("stt %.2fs (%.1fs audio): %s", time.perf_counter() - t0, len(audio) / 16000, text)
        return text.strip()

    def warm_up(self):
        self(np.zeros(16000, dtype=np.float32))

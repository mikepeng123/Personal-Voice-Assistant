"""Wake word matching.

The assistant only acts when it is addressed: the wake word must be the first word of a
sentence, optionally after a filler like "okay" or "hey". So these work:
    "Okay, loop, book a dentist appointment"
    "... tomorrow at 3. Loop. Book a dentist appointment"
and this is ignored:
    "I like to loop videos"
Speech-to-text often mishears the name, so a list of spellings is accepted (config.toml).
"""
import re

FILLERS = {"okay", "ok", "hey", "hi", "hello", "so", "um", "uh", "oh", "yo", "alright", "right", "and"}
WORD = re.compile(r"[A-Za-z']+")
BOUNDARY = re.compile(r"[.!?]+")


def find_wake(text: str, wake_words: set[str]) -> str | None:
    """Return the command after the wake word ('' if only the wake word was said),
    or None if the text does not address the assistant."""
    starts = [0] + [m.end() for m in BOUNDARY.finditer(text)]
    for pos in starts:
        while True:
            m = WORD.search(text, pos)
            if not m or BOUNDARY.search(text, pos, m.start()):
                break  # sentence ended before any real word
            word = m.group().lower().strip("'")
            if word in wake_words:
                return text[m.end():].lstrip(" ,.!?;:-").strip()
            if word not in FILLERS:
                break
            pos = m.end()
    return None


def normalize(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))

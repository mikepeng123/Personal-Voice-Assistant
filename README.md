# Loop: a local voice assistant

Everything runs locally with Ollama. All settings are in `config.toml`.

## Setup (Stage 0)

```bash
# 1. Models
ollama pull gemma4:12b-mlx        # or any tool-calling model; set chat_model in config.toml
ollama pull nomic-embed-text

# 2. Memory store (Qdrant server, local)
docker run -d -p 6333:6333 -v "$(pwd)/data/qdrant:/qdrant/storage" qdrant/qdrant

# 3. Python packages (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Check
python check_setup.py
```

Microphone access (needed from Stage 3): on macOS allow your terminal or IDE under
System Settings > Privacy & Security > Microphone.

## Run

```bash
python cli.py            # typed agent; /list shows the database, /reset clears the chat
python check_stage1.py   # runs the eight Stage 1 checks against a fresh database
python check_stage2.py   # time to first audio, and stop latency
python voice.py          # always-on voice assistant (you can also type commands)
python voice.py --levels # microphone level meter, for tuning [audio] in config.toml
python check_stage3.py   # speak the eight Stage 1 commands; graded automatically
python check_stage4.py   # remember -> restart -> recall (runs two processes)
streamlit run app.py     # the web page: status, conversation, tool calls, memory, appointments
```

The first run of `voice.py` downloads the Whisper model (about 1.6 GB) once; after that it runs offline.

## Files

| File | Purpose |
|---|---|
| `config.toml` | models, names, paths |
| `config.py` | loads the config, sets up rotating log file |
| `appointments.py` | SQLite appointment store |
| `tools.py` | the four tools, their schemas, and safety checks in code |
| `agent.py` | system prompt (with current date) and tool-calling loop |
| `cli.py` | typed chat for testing (spoken replies) |
| `speech_out.py` | sentence splitting, cleanup, background speaker with instant stop |
| `audio_in.py` | microphone, loudness-based utterance segmentation, speech-to-text |
| `wake.py` | wake word matching |
| `memory.py` | long-term memory: facts with embeddings in Qdrant, search by meaning |
| `assistant.py` | the always-on loop (threads, mute while speaking, follow-up window) |
| `voice.py` | terminal front end for the voice loop |
| `app.py` | Streamlit page; the assistant is created once with `st.cache_resource` and runs in its own threads |
| `check_*.py`, `diagnose_speed.py` | stage checks and speed diagnosis |

## Design notes

Rules that matter most are enforced in code as well as in the prompt, because small models
sometimes ignore the prompt: a location the user never said is rejected, `delete_appointment`
only runs when the user's latest message is a yes, and an identical second booking returns the
existing id instead of creating a duplicate.

Speed: the system prompt holds only the date (the time of day is a `[HH:MM]` tag on each user
message), so the prompt and tool definitions stay identical all day and Ollama reuses its cached
processing of them; `warm_up()` fills that cache at start-up. Hidden reasoning is off
(`think = false`). With `fast_confirmations = true`, a successful create, update or delete is
confirmed from the tool result in code instead of a second model call. This was a deliberate
trade-off to get the first spoken words under 4 seconds on a 12B model: the confirmation wording
is fixed, but it can only ever describe what the tool actually did. Everything else (questions,
listing, confirming deletes, errors) is still written by the model.

Memory: instead of mem0 I wrote a small memory module on Qdrant (`memory.py`). The reason is
speed. mem0 calls the chat model with its own prompt, which replaces Ollama's cached system prompt
and conversation, so the next command would pay several seconds of prompt processing again. Here
fact extraction is asked as one extra message after the real conversation, so everything before it
is still cached. Before each request the closest memories are added to the end of the user message
(not the system prompt, which must stay identical). A new fact that is nearly the same as an old
one replaces it. Appointments are kept out twice: the extraction prompt excludes them, and a code
filter drops any fact that names an event together with a date or time.

## Results

| Stage | Check | Result |
|---|---|---|
| 0 | Models answer, stores accept a write and read | 5 of 5 passed |
| 1 | Eight typed commands | 8 of 8 passed |
| 2 | First audio under 4 s, stop under 0.5 s | Passed (stop silenced speech in 0.021 s) |
| 3 | Eight spoken commands, at least 5 | 8 of 8 passed (after the fixes below) |
| 4 | Remember, restart, recall | Passed (similarity of the stored fact to the question: 0.806) |
| 5 | Status moves through its stages on the page | Passed |

Stage 2 took three rounds. The first booking command started speaking after 15.97 s: the model
was producing hidden reasoning, and the timestamp at the top of the system prompt stopped Ollama
from reusing its cached prompt processing. Turning reasoning off and keeping the prompt prefix
stable brought it to 4.81 s; confirming successful bookings from the tool result in code
(instead of a second model call) brought it under 4 s.

## Stage 3: speech-to-text failures

The first spoken run passed only 1 of 8 checks. The failures below were caused by the
microphone still being muted (reply tone and echo delay) when the next prompt appeared, so the
wake word and first syllables were cut off, and by utterances split at short pauses. After the
fixes (announce "listening" only once the microphone is open, keep it open after a bare wake
word, join utterances split by a short pause, keep 0.5 s of audio before speech starts) a second
run passed 8 of 8.

| Said | Speech-to-text produced | Reply |
|---|---|---|
| Loop, book a dentist appointment tomorrow at 3 pm in Chicago | Loop. Look at dentist appointment tomorrow at 3 p.m. in Chicago. | I couldn't find any appointments for tomorrow at 3 p.m. Would you like me to book one for you? |
| Loop, book a haircut on Friday at 10 am | Book a haircut on Friday at 10 a.m. | Where would you like to have your haircut? (failed only because step 1 had saved nothing) |
| Loop, what is on my calendar | That is on my calendar. | I'm sorry, I don't see a haircut on your calendar for Friday at 10 a.m. |
| Loop, move my dentist appointment to 4 pm | Move my dentist appointment to 4pm. | I couldn't find any dentist appointments in your calendar. |
| Loop, delete my dentist appointment | Loop. Delete. / My dentist appointment. (split in two) | I'm sorry, I don't have any appointments to delete. |
| Loop, book lunch with Sam on Friday at noon at Cafe Roma | Loop. / Cafe Roma. (rest cut off) | I couldn't find any appointments for Cafe Roma. |

## Stage 6: measurements

| Model | Checks passed (of 8) | Total seconds for the eight checks |
|---|---|---|
| gemma4:12b-mlx | 8 | 24.7 |
| qwen3:8b | 8 | 18.0 |

Both models passed every check; qwen3:8b was about 27% faster, as expected from a smaller model.
The checks measure the model together with the code-level guards (location must come from the
user, deletes need a yes, duplicates are merged), which is part of why both pass.

My biggest source of mistakes was the voice pipeline's timing rather than the model: in the first
spoken run only 1 of 8 checks passed, because the prompt to speak appeared while the microphone
was still muted for the reply tone, so the wake word and first syllables were cut off. I fixed it
by announcing "listening" only after the microphone had really reopened, keeping the microphone
open after a bare wake word, joining utterances split by a short pause, and keeping more audio
from before speech starts, after which all 8 spoken checks passed.

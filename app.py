"""Stage 5: the web page.

    streamlit run app.py

Streamlit reruns this whole script on every click. The assistant (models, microphone, threads)
is created once with st.cache_resource and keeps running in its own threads; the page only
reads its state and calls submit_text / set_muted / stop_speaking.
"""
import html
import json
import re

import streamlit as st

import config
from agent import Agent
from appointments import AppointmentStore
from assistant import VoiceAssistant
from audio_in import Transcriber
from memory import MemoryStore
from speech_out import Speaker
from tools import spoken_when

st.set_page_config(page_title="Loop", page_icon="🎙️", layout="wide")


@st.cache_resource
def get_assistant():
    cfg = config.load()
    config.setup_logging(cfg)
    store = AppointmentStore(cfg["paths"]["appointments_db"])
    memory = MemoryStore(cfg)
    agent = Agent(cfg, store, memory=memory)
    va = VoiceAssistant(cfg, agent, Speaker(cfg["speech"]["tts_command"]), Transcriber(cfg["stt"]))
    va.start()  # background threads, started exactly once
    return va, store, memory, cfg


va, store, memory, cfg = get_assistant()
NAME = cfg["assistant"]["name"]

# ---------- look ----------
# One loud element: the status banner. Everything else stays plain Streamlit.
st.markdown("""
<style>
.status { border-radius: 14px; padding: 1.1rem 1.4rem; color: #fff; margin-bottom: .4rem; }
.status h2 { margin: 0; font-size: 2.1rem; line-height: 1.15; color: #fff; }
.status p { margin: .25rem 0 0; font-size: 1.05rem; opacity: .92; }
.status.listening { animation: breathe 2.4s ease-in-out infinite; }
@keyframes breathe { 50% { filter: brightness(1.12); } }
@media (prefers-reduced-motion: reduce) { .status.listening { animation: none; } }
.history { font-size: .85rem; opacity: .75; }
</style>
""", unsafe_allow_html=True)

STATUS = {
    "loading":   ("#5f6b7a", "⏳", "Loading", "Models and microphone are starting. It is not listening yet."),
    "listening": ("#15803d", "🎙️", "Listening", f"Say “{NAME}, …” followed by what you want."),
    "thinking":  ("#b45309", "💭", "Thinking", "Working on your request. The microphone is off."),
    "speaking":  ("#1d4ed8", "🔊", "Speaking", "The microphone is off while it talks."),
    "muted":     ("#b91c1c", "🔇", "Muted", "Not listening. Turn off mute in the sidebar to continue."),
}


def md(text: str) -> str:
    """Escape user text so Markdown characters show as typed."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~])", r"\\\1", text or "")


# ---------- sidebar: controls ----------
with st.sidebar:
    st.header("Controls")
    st.toggle("Mute microphone", key="mute", value=va.user_muted,
              on_change=lambda: va.set_muted(st.session_state.mute))
    if st.button("Stop speaking", use_container_width=True):
        va.stop_speaking()
    if st.button("Start a new conversation", use_container_width=True,
                 help="Forgets the current conversation. Memories and appointments stay."):
        va.agent.reset()
    st.divider()
    st.caption(f"Chat model: {cfg['models']['chat_model']}")
    st.caption(f"Speech-to-text: {cfg['stt']['model']}")
    st.caption(f"Wake word: {NAME} (also accepts {', '.join(cfg['wake']['words'][1:5])}, …)")
    st.caption(f"After a reply, no wake word is needed for {cfg['wake']['follow_up_s']} seconds.")


# ---------- status banner ----------
@st.fragment(run_every=0.3)
def status_banner():
    status = va.status
    color, icon, title, hint = STATUS[status]
    if status == "listening" and va.follow_up_left > 0:
        hint = f"Go ahead, no need to say “{NAME}” for {va.follow_up_left:.0f} more seconds."
    st.markdown(f'<div class="status {status}" style="background:{color}">'
                f'<h2>{icon} {title}</h2><p>{html.escape(hint)}</p></div>', unsafe_allow_html=True)
    if va.status_log:
        trail = "  ›  ".join(f"{s} {t}" for t, s in va.status_log)
        st.markdown(f'<div class="history">Recent status: {html.escape(trail)}</div>',
                    unsafe_allow_html=True)


status_banner()
st.write("")
left, right = st.columns([3, 2], gap="large")


# ---------- conversation ----------
def render_turn(turn: dict, live: bool = False):
    with st.container(border=True):
        when = turn["time"] + (f", {turn['seconds']} s" if turn.get("seconds") else "")
        st.caption(("Now: " if live else "") + when)
        verb = "You typed" if turn.get("typed") else "Heard"
        st.markdown(f"**{verb}:** {md(turn['heard'])}")
        if not turn.get("typed") and turn["command"] != turn["heard"]:
            st.caption(f"Command after the wake word: {turn['command']}")
        if turn.get("memories"):
            st.caption("Memories used: " + " ".join(turn["memories"]))
        tools = turn["tools"]
        if tools:
            with st.expander(f"{len(tools)} tool call{'s' if len(tools) > 1 else ''}"):
                for call in tools:
                    st.code(f"{call['name']}({json.dumps(call['args'])})\n{call['result']}",
                            language=None, wrap_lines=True)
        reply = turn["reply"] or ("…" if live else "(no reply)")
        st.markdown(f"**{NAME}:** {md(reply)}")


with left:
    st.subheader("Conversation")

    @st.fragment(run_every=1.0)
    def conversation():
        current, turns = va.current, list(va.turns)
        if not current and not turns:
            st.info(f"Nothing yet. Say “{NAME}, what is on my calendar?” or type a command below.")
        if current:
            render_turn(current, live=True)
        for turn in reversed(turns[-15:]):
            render_turn(turn)

    conversation()


# ---------- appointments, memory, recent speech ----------
with right:
    @st.fragment(run_every=2.0)
    def side_panels():
        st.subheader("Appointments")
        appts = store.list_all()
        if not appts:
            st.caption("No appointments.")
        for a in appts:
            st.markdown(f"**{md(a['title'])}**, {spoken_when(a['date'])}, {md(a['location'])}  \n"
                        f"<span style='opacity:.6'>id {a['id']}, {a['date']}</span>",
                        unsafe_allow_html=True)

        st.subheader("Memory")
        try:
            items = memory.all()
        except Exception as e:
            items = []
            st.error(f"Memory store unreachable: {e}. Is the Qdrant container running?")
        if not items:
            st.caption(f"Nothing remembered yet. Try “{NAME}, remember that I like tea.”")
        for m in items:
            st.markdown(f"- {md(m['text'])}")

        st.subheader("Recent speech")
        icons = {"heard": "👂", "ignored": "🚫", "wake": "👋", "joined": "🔗", "error": "⚠️"}
        notes = {"ignored": " (no wake word)", "wake": " (wake word only)", "joined": " (joined pieces)"}
        events = list(va.events)[-8:]
        if not events:
            st.caption("Everything the microphone hears shows up here, including speech it ignores.")
        for t, kind, text in reversed(events):
            st.markdown(f"{icons.get(kind, '')} {md(text)}<span style='opacity:.6'>{notes.get(kind, '')}, {t}</span>",
                        unsafe_allow_html=True)

    side_panels()


# ---------- typed command ----------
if text := st.chat_input(f"Type a command for {NAME}"):
    va.submit_text(text)
    st.toast("Sent")

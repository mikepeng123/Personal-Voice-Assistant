"""The four appointment tools: JSON schemas for the model, plus the code that runs them.

The prompt asks the model to behave, but small models do not always listen, so the
most important rules are also enforced here in code:
  - a location the user never said is rejected (no invented locations)
  - delete only runs if the user's latest message is a "yes"
  - booking the exact same appointment twice returns the existing id
"""
import json
import re
from datetime import datetime

from appointments import AppointmentStore

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "create_appointment",
            "description": "Save one new appointment. Returns its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Start time as YYYY-MM-DD HH:MM, 24-hour clock"},
                    "location": {"type": "string", "description": "Where it is, exactly as the user said it"},
                    "title": {"type": "string", "description": "Short title, e.g. 'Dentist'"},
                    "description": {"type": "string", "description": "Optional extra details"},
                },
                "required": ["date", "location", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_appointments",
            "description": "Return all appointments ordered by date, with their ids.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_appointment",
            "description": "Change an existing appointment. Only the fields given are changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer", "description": "Id from list_appointments"},
                    "date": {"type": "string", "description": "New start time as YYYY-MM-DD HH:MM"},
                    "location": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_appointment",
            "description": "Delete one appointment. Only call after the user said yes to deleting it.",
            "parameters": {
                "type": "object",
                "properties": {"appointment_id": {"type": "integer", "description": "Id from list_appointments"}},
                "required": ["appointment_id"],
            },
        },
    },
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}

REMEMBER_SPEC = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": "Save a lasting personal fact or preference about the user, when they ask you to "
                       "remember something. Never use it for appointments.",
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string",
                                    "description": "One short sentence, e.g. 'The user prefers morning meetings.'"}},
            "required": ["fact"],
        },
    },
}

EVENT_WORDS = re.compile(r"\b(appointments?|meetings?|booking|booked|reservations?|lunch|dinner|breakfast|"
                         r"call|haircut|dentist)\b", re.I)
WHEN_WORDS = re.compile(r"\b(?:today|tomorrow|tonight|yesterday|next (?:week|month)|monday|tuesday|wednesday|"
                        r"thursday|friday|saturday|sunday|noon|o'clock)\b"
                        r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\d{4}-\d{2}-\d{2}", re.I)


def looks_like_appointment(text: str) -> bool:
    """'Lunch with Sam on Friday at noon' is an appointment; 'prefers morning meetings' is not."""
    return bool(EVENT_WORDS.search(text) and WHEN_WORDS.search(text))

YES = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|do it|go ahead|please do)\b", re.I)


def _ok(message, **data):
    return json.dumps({"ok": True, "message": message, **data}, separators=(",", ":"))


def _err(message):
    return json.dumps({"ok": False, "error": message}, separators=(",", ":"))


def _said_by_user(location: str, user_text: str) -> bool:
    """True if the location (or a word of it) appears in what the user typed."""
    words = re.findall(r"[a-z0-9]+", location.lower())
    long_words = [w for w in words if len(w) >= 3] or words
    return bool(long_words) and any(w in user_text for w in long_words)


def _id(args):
    if "appointment_id" not in args:
        raise ValueError("appointment_id is required; call list_appointments to find it")
    return int(args["appointment_id"])


class ToolRunner:
    def __init__(self, store: AppointmentStore, memory=None):
        self.store = store
        self.memory = memory
        self.specs = TOOL_SPECS + ([REMEMBER_SPEC] if memory else [])
        self.names = {t["function"]["name"] for t in self.specs}

    def run(self, name: str, args: dict, user_texts: list[str]) -> str:
        """Run one tool. user_texts are the user's messages in this conversation, newest last."""
        if name not in self.names:
            return _err(f"unknown tool {name!r}")
        try:
            return getattr(self, f"_{name}")(args or {}, user_texts)
        except (ValueError, TypeError) as e:
            return _err(str(e))

    def _remember(self, args, user_texts):
        fact = str(args.get("fact") or "").strip()
        if not fact:
            return _err("fact is required")
        if looks_like_appointment(fact):
            return _err("That is an appointment, not a personal fact. Use create_appointment for it.")
        result = self.memory.add(fact, source="tool")
        return _ok(f"memory {result['action']}", memory=result["text"], action=result["action"])

    def _create_appointment(self, args, user_texts):
        location = str(args.get("location") or "").strip()
        if not location or not _said_by_user(location, " ".join(user_texts).lower()):
            return _err("The user has not said where. Ask them for the location in one short question. Do not guess.")
        if not args.get("date"):
            return _err("date is required as YYYY-MM-DD HH:MM")
        title = str(args.get("title") or "Appointment").strip()
        date = self.store.normalize_date(args["date"])
        for a in self.store.list_all():  # same booking twice in one turn -> keep one
            if a["date"] == date and a["title"].lower() == title.lower() and a["location"].lower() == location.lower():
                return _ok(f"already booked as appointment id {a['id']}", appointment=a)
        appt = self.store.create(date, location, title, args.get("description") or "")
        return _ok(f"created appointment id {appt['id']}", appointment=appt)

    def _list_appointments(self, args, user_texts):
        appts = self.store.list_all()
        return _ok(f"{len(appts)} appointment(s)", appointments=appts)

    def _update_appointment(self, args, user_texts):
        appointment_id = _id(args)
        if not self.store.get(appointment_id):
            return _err(f"no appointment with id {appointment_id}; call list_appointments")
        location = args.get("location")
        if location and not _said_by_user(str(location), " ".join(user_texts).lower()):
            return _err("The user has not given a new location. Do not change it.")
        changes = {k: args.get(k) for k in ("date", "location", "title", "description")}
        appt = self.store.update(appointment_id, **changes)
        return _ok(f"updated appointment id {appointment_id}", appointment=appt)

    def _delete_appointment(self, args, user_texts):
        appointment_id = _id(args)
        appt = self.store.get(appointment_id)
        if not appt:
            return _err(f"no appointment with id {appointment_id}; call list_appointments")
        if not user_texts or not YES.match(user_texts[-1]):
            return _err(
                f"Not confirmed. Ask the user: delete {appt['title']} on {appt['date']} "
                "in " + appt["location"] + "? Only delete after they answer yes."
            )
        self.store.delete(appointment_id)
        return _ok(f"deleted appointment id {appointment_id}", appointment=appt)


# ---------- spoken confirmations built from tool results ----------

WRITE_TOOLS = {"create_appointment", "update_appointment", "delete_appointment", "remember"}


def spoken_when(date_str: str, now: datetime | None = None) -> str:
    """'2026-11-16 15:00' -> 'tomorrow at 3 PM' (or 'on Friday at noon', 'on December 3 at 9:30 AM')."""
    now = now or datetime.now()
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    days = (dt.date() - now.date()).days
    if days == 0:
        day = "today"
    elif days == 1:
        day = "tomorrow"
    elif 1 < days < 7:
        day = f"on {dt:%A}"
    else:
        day = f"on {dt:%B} {dt.day}"
    if dt.hour == 12 and dt.minute == 0:
        clock = "noon"
    else:
        clock = str(dt.hour % 12 or 12) + (f":{dt.minute:02d}" if dt.minute else "")
        clock += " AM" if dt.hour < 12 else " PM"
    return f"{day} at {clock}"


def confirmation(results: list[tuple[str, str]], now: datetime | None = None) -> str | None:
    """If every tool call in a step was a successful create/update/delete, return the spoken
    confirmation built from the tool results. Otherwise None, and the model writes the reply."""
    if not results or any(name not in WRITE_TOOLS for name, _ in results):
        return None
    sentences, seen = [], set()
    for name, raw in results:
        data = json.loads(raw)
        if not data.get("ok"):
            return None
        if name == "remember":
            sentences.append({"duplicate": "I already know that.", "updated": "Got it, I have updated that."}
                             .get(data.get("action"), "Got it, I will remember that."))
            continue
        a = data["appointment"]
        if (name, a["id"]) in seen:
            continue  # the model sent the same call twice; say it once
        seen.add((name, a["id"]))
        title = a["title"][:1].upper() + a["title"][1:]
        when = spoken_when(a["date"], now)
        if name == "create_appointment" and data["message"].startswith("already"):
            sentences.append(f"That is already booked, the id is {a['id']}.")
        elif name == "create_appointment":
            sentences.append(f"Booked, the id is {a['id']}. {title} {when} in {a['location']}.")
        elif name == "update_appointment":
            sentences.append(f"Updated appointment {a['id']}. {title} is now {when} in {a['location']}.")
        else:
            sentences.append(f"Deleted. {title} {when} is cancelled.")
    return " ".join(sentences)

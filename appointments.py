"""SQLite store for appointments."""
import sqlite3
from contextlib import closing
from datetime import datetime

DATE_FMT = "%Y-%m-%d %H:%M"
FIELDS = ("date", "location", "title", "description")


class AppointmentStore:
    def __init__(self, path: str):
        self.path = path
        self._exec(
            """CREATE TABLE IF NOT EXISTS appointments (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   date TEXT NOT NULL,
                   location TEXT NOT NULL,
                   title TEXT NOT NULL,
                   description TEXT NOT NULL DEFAULT '')"""
        )

    def _exec(self, sql, params=()):
        # A short-lived connection per call keeps this safe to use from
        # several threads later (listener thread + web page in Stage 5).
        with closing(sqlite3.connect(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            conn.commit()
            return rows, cur.lastrowid, cur.rowcount

    @staticmethod
    def normalize_date(value: str) -> str:
        """Raise ValueError unless value is 'YYYY-MM-DD HH:MM'."""
        try:
            return datetime.strptime(str(value).strip(), DATE_FMT).strftime(DATE_FMT)
        except ValueError:
            raise ValueError(f"date must look like YYYY-MM-DD HH:MM, got {value!r}") from None

    def create(self, date, location, title, description="") -> dict:
        _, new_id, _ = self._exec(
            "INSERT INTO appointments (date, location, title, description) VALUES (?, ?, ?, ?)",
            (self.normalize_date(date), location.strip(), title.strip(), (description or "").strip()),
        )
        return self.get(new_id)

    def get(self, appointment_id: int) -> dict | None:
        rows, _, _ = self._exec("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
        return rows[0] if rows else None

    def list_all(self) -> list[dict]:
        rows, _, _ = self._exec("SELECT * FROM appointments ORDER BY date, id")
        return rows

    def update(self, appointment_id: int, **changes) -> dict | None:
        changes = {k: v for k, v in changes.items() if k in FIELDS and v not in (None, "")}
        if "date" in changes:
            changes["date"] = self.normalize_date(changes["date"])
        if changes:
            sets = ", ".join(f"{k} = ?" for k in changes)
            self._exec(f"UPDATE appointments SET {sets} WHERE id = ?", (*changes.values(), appointment_id))
        return self.get(appointment_id)

    def delete(self, appointment_id: int) -> bool:
        _, _, count = self._exec("DELETE FROM appointments WHERE id = ?", (appointment_id,))
        return count > 0

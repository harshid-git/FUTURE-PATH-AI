"""Storage and advisor logic for Futurepath AI (no web-framework imports)."""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from data import CAREERS, SECTORS

log = logging.getLogger("futurepath")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("FUTUREPATH_DB", str(BASE_DIR / "futurepath.db"))
MAX_ADVICE = 20  # the frontend keeps only the 20 most recent pieces of advice
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SECTOR_BY_ID = {s["id"]: s for s in SECTORS}
CAREER_BY_ID = {c["id"]: c for c in CAREERS}

try:  # dates shown to the student ("18 Nov") use this timezone
    from zoneinfo import ZoneInfo

    DISPLAY_TZ = ZoneInfo(os.getenv("APP_TZ", "Asia/Kolkata"))
except Exception:  # tzdata missing (e.g. some Windows installs)
    DISPLAY_TZ = timezone.utc


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db() -> None:
    """Create tables and, on first run only, seed the same demo data the frontend ships with."""
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profile (
                id        INTEGER PRIMARY KEY CHECK (id = 1),
                name      TEXT NOT NULL,
                year      TEXT NOT NULL,
                subjects  TEXT NOT NULL DEFAULT '',
                location  TEXT NOT NULL DEFAULT '',
                interests TEXT NOT NULL DEFAULT '',
                sectors   TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS saved (
                career_id  TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS advice (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        if conn.execute("SELECT 1 FROM profile").fetchone() is None:
            conn.execute(
                "INSERT INTO profile (id, name, year, subjects, location, interests, sectors) "
                "VALUES (1, ?, ?, ?, '', '', ?)",
                ("Aarav Mehta", "Class 12", "History, Political Science", json.dumps(["civil", "science"])),
            )
            for cid in ("ias", "isro", "ifs"):
                conn.execute("INSERT INTO saved (career_id, created_at) VALUES (?, ?)", (cid, _now()))
            # inserted oldest-first so that ORDER BY id DESC returns newest-first
            conn.execute(
                "INSERT INTO advice (text, created_at) VALUES (?, ?)",
                (
                    "Since you enjoy science, take a look at ISRO and DRDO careers too. These have different "
                    "degree requirements, so map out subject choices early and keep both paths open while you explore.",
                    "2026-11-16T09:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO advice (text, created_at) VALUES (?, ?)",
                (
                    "Your interest in public affairs and history could make civil services a rewarding path to "
                    "explore. Start by learning what the UPSC CSE syllabus covers, then compare roles such as IAS, "
                    "IFS and IRS to see which kind of impact feels right to you.",
                    "2026-11-18T09:00:00+00:00",
                ),
            )


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
def get_profile(conn: sqlite3.Connection) -> dict:
    row = dict(conn.execute("SELECT * FROM profile WHERE id = 1").fetchone())
    row.pop("id")
    row["sectors"] = json.loads(row["sectors"])
    return row


def save_profile(conn: sqlite3.Connection, p: dict) -> dict:
    conn.execute(
        "UPDATE profile SET name=?, year=?, subjects=?, location=?, interests=?, sectors=? WHERE id = 1",
        (p["name"], p["year"], p["subjects"], p["location"], p["interests"], json.dumps(p["sectors"])),
    )
    return get_profile(conn)


def profile_completeness(p: dict) -> int:
    checks = [
        p["name"] and p["name"] != "Student",
        p["year"],
        p["subjects"].strip(),
        p["location"].strip(),
        p["interests"].strip(),
        p["sectors"],
    ]
    return round(100 * sum(bool(c) for c in checks) / len(checks))


# --------------------------------------------------------------------------- #
# Careers, saved careers, advice
# --------------------------------------------------------------------------- #
def saved_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT career_id FROM saved ORDER BY created_at, rowid").fetchall()
    return [r["career_id"] for r in rows]


def career_out(c: dict, saved: set[str]) -> dict:
    return {**c, "sector_name": SECTOR_BY_ID[c["sector"]]["name"], "saved": c["id"] in saved}


def search_careers(sector: str | None, q: str | None) -> list[dict]:
    term = (q or "").strip().lower()
    out = []
    for c in CAREERS:
        if sector and sector != "all" and c["sector"] != sector:
            continue
        haystack = f"{c['title']}{c['exam']}{c['desc']}{SECTOR_BY_ID[c['sector']]['name']}".lower()
        if term and term not in haystack:
            continue
        out.append(c)
    return out


def recommendations(profile: dict, limit: int = 3) -> list[dict]:
    rec = [c for c in CAREERS if c["sector"] in profile["sectors"]][:limit]
    return rec or CAREERS[:limit]


def fmt_advice(row: sqlite3.Row) -> dict:
    dt = datetime.fromisoformat(row["created_at"]).astimezone(DISPLAY_TZ)
    return {
        "id": row["id"],
        "text": row["text"],
        "date": f"{dt.day} {dt.strftime('%b')}",  # e.g. "18 Nov", as the frontend displays it
        "created_at": row["created_at"],
    }


def list_advice(conn: sqlite3.Connection) -> list[dict]:
    return [fmt_advice(r) for r in conn.execute("SELECT * FROM advice ORDER BY id DESC")]


def add_advice(conn: sqlite3.Connection, text: str) -> dict:
    cur = conn.execute("INSERT INTO advice (text, created_at) VALUES (?, ?)", (text, _now()))
    conn.execute(
        "DELETE FROM advice WHERE id NOT IN (SELECT id FROM advice ORDER BY id DESC LIMIT ?)",
        (MAX_ADVICE,),
    )
    return fmt_advice(conn.execute("SELECT * FROM advice WHERE id = ?", (cur.lastrowid,)).fetchone())


# --------------------------------------------------------------------------- #
# Advisor
# --------------------------------------------------------------------------- #
# Extra words that point to a sector, beyond the first word of its name.
SECTOR_KEYWORDS = {
    "civil": ["upsc", "ias", "ifs", "irs", "administrat", "government", "policy", "diplomac", "collector"],
    "defence": ["army", "navy", "air force", "military", "nda", "officer", "soldier", "pilot"],
    "banking": ["bank", "finance", "economics", "economy", "rbi", "money", "accountan"],
    "science": ["scien", "space", "isro", "drdo", "research", "physics", "chemistry", "technology"],
    "engineering": ["engineer", "construction", "infrastructure", "mechanical", "civil engineering", "electrical"],
    "health": ["doctor", "medic", "neet", "hospital", "biology", "nurse"],
    "education": ["teach", "school", "ctet", "professor", "student learning"],
    "law": ["lawyer", "judge", "judicial", "court", "justice", "llb", "legal"],
}


def _sector_hits(text: str) -> set[str]:
    text = text.lower()
    hits = set()
    for s in SECTORS:
        words = [s["name"].lower().split()[0], *SECTOR_KEYWORDS.get(s["id"], [])]
        if any(re.search(r"\b" + re.escape(w), text) for w in words):
            hits.add(s["id"])
    return hits


def suggest_careers(message: str, profile: dict, limit: int = 3) -> list[dict]:
    """Careers that fit the question first, then the student's profile; best match first."""
    from_question = _sector_hits(message)
    from_profile = _sector_hits(f"{profile['interests']} {profile['subjects']}") | set(profile["sectors"])
    by_match = sorted(CAREERS, key=lambda c: -c["match"])
    ordered = [c for c in by_match if c["sector"] in from_question]
    ordered += [c for c in by_match if c["sector"] in from_profile and c["sector"] not in from_question]
    return (ordered or by_match)[:limit]


def rule_based_reply(message: str, profile: dict) -> str:
    matches = suggest_careers(message, profile)
    subjects = f" — especially your interest in {profile['subjects']}" if profile["subjects"] else ""
    items = "\n\n".join(
        f"• {c['title']} ({c['match']}% interest match)\n  {c['desc']}\n  Exam: {c['exam']}\n  Eligibility: {c['eligibility']}"
        for c in matches
    )
    return (
        f"From what you’ve shared{subjects} — here are a few paths worth exploring:\n\n{items}\n\n"
        "A good next step is to compare the subjects and preparation each path needs. "
        "Which one would you like to learn more about?"
    )


def llm_reply(message: str, history: list[dict], profile: dict) -> str | None:
    """Ask Claude for advice. Returns None if no API key / SDK, or on any error (caller falls back)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("ANTHROPIC_API_KEY is set but the 'anthropic' package is not installed")
        return None

    catalogue = "\n".join(
        f"- {c['title']} [{SECTOR_BY_ID[c['sector']]['name']}] exam: {c['exam']}; eligibility: {c['eligibility']}"
        for c in CAREERS
    )
    system = (
        "You are the Futurepath Guide, a warm, practical career advisor for Indian students exploring "
        "government and public-service careers.\n"
        f"Student: {profile['name']}, {profile['year']}. Subjects: {profile['subjects'] or 'not shared'}. "
        f"Interests: {profile['interests'] or 'not shared'}. Location: {profile['location'] or 'not shared'}.\n\n"
        f"Careers in the app:\n{catalogue}\n\n"
        "Rules: reply in plain text only (no markdown), under 180 words. Prefer the careers above and be "
        "honest that exam patterns, age limits and eligibility change, so the student should confirm them on "
        "the official notification. End with one question that helps the student take a next step."
    )
    messages = [{"role": t["role"], "content": t["content"]} for t in history[-10:]]
    messages.append({"role": "user", "content": message})
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=700, system=system, messages=messages
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
    except Exception as exc:  # network, auth, bad history shape, ...
        log.warning("Claude call failed, using rule-based advisor: %s", exc)
        return None

"""Futurepath AI — backend API.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

import core
from data import CAREERS, SECTORS


@asynccontextmanager
async def lifespan(_: FastAPI):
    core.init_db()
    yield


app = FastAPI(title="Futurepath AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ProfileIn(BaseModel):
    name: str = Field("Student", max_length=100)
    year: str = Field("Class 12", max_length=40)
    subjects: str = Field("", max_length=300)
    location: str = Field("", max_length=100)
    interests: str = Field("", max_length=1000)
    sectors: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return v.strip() or "Student"  # same fallback the frontend uses

    @field_validator("subjects", "location", "interests", "year")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("sectors")
    @classmethod
    def _sectors(cls, v: list[str]) -> list[str]:
        unknown = [s for s in v if s not in core.SECTOR_BY_ID]
        if unknown:
            raise ValueError(f"Unknown sector id(s): {', '.join(unknown)}")
        return list(dict.fromkeys(v))  # de-duplicate, keep order


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list)
    save_advice: bool = True  # the frontend saves every advisor reply to "Saved advice"

    @field_validator("message")
    @classmethod
    def _msg(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sectors")
def list_sectors():
    return SECTORS


@app.get("/api/careers")
def list_careers(
    sector: str | None = Query(None, description="Sector id, or 'all'"),
    q: str | None = Query(None, description="Search title, exam, description or sector name"),
):
    if sector and sector != "all" and sector not in core.SECTOR_BY_ID:
        raise HTTPException(404, f"Unknown sector '{sector}'")
    with core.db() as conn:
        saved = set(core.saved_ids(conn))
    return [core.career_out(c, saved) for c in core.search_careers(sector, q)]


@app.get("/api/careers/{career_id}")
def get_career(career_id: str):
    career = core.CAREER_BY_ID.get(career_id)
    if not career:
        raise HTTPException(404, "Career not found")
    with core.db() as conn:
        saved = set(core.saved_ids(conn))
    return core.career_out(career, saved)


# --------------------------------------------------------------------------- #
# Home dashboard
# --------------------------------------------------------------------------- #
@app.get("/api/dashboard")
def dashboard():
    with core.db() as conn:
        profile = core.get_profile(conn)
        saved = core.saved_ids(conn)
        advice_count = conn.execute("SELECT COUNT(*) FROM advice").fetchone()[0]
    first_name = profile["name"].split()[0] if profile["name"].split() else "Student"
    return {
        "name": profile["name"],
        "first_name": first_name,
        "initial": first_name[0].upper(),
        "saved_count": len(saved),
        "advice_count": advice_count,
        "profile_completeness": core.profile_completeness(profile),
        "recommendations": [core.career_out(c, set(saved)) for c in core.recommendations(profile)],
    }


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
@app.get("/api/profile")
def read_profile():
    with core.db() as conn:
        profile = core.get_profile(conn)
        return {**profile, "saved_count": len(core.saved_ids(conn)),
                "advice_count": conn.execute("SELECT COUNT(*) FROM advice").fetchone()[0],
                "completeness": core.profile_completeness(profile)}


@app.put("/api/profile")
def update_profile(body: ProfileIn):
    with core.db() as conn:
        profile = core.save_profile(conn, body.model_dump())
        return {**profile, "completeness": core.profile_completeness(profile)}


@app.put("/api/profile/sectors/{sector_id}")
def toggle_profile_sector(sector_id: str):
    """Add the sector to the student's interests, or remove it if already there."""
    if sector_id not in core.SECTOR_BY_ID:
        raise HTTPException(404, "Unknown sector")
    with core.db() as conn:
        profile = core.get_profile(conn)
        sectors = profile["sectors"]
        profile["sectors"] = [s for s in sectors if s != sector_id] if sector_id in sectors else [*sectors, sector_id]
        profile = core.save_profile(conn, profile)
        return {**profile, "recommendations": [core.career_out(c, set(core.saved_ids(conn)))
                                                for c in core.recommendations(profile)]}


# --------------------------------------------------------------------------- #
# Saved careers
# --------------------------------------------------------------------------- #
@app.get("/api/saved")
def list_saved():
    with core.db() as conn:
        ids = core.saved_ids(conn)
    return [core.career_out(core.CAREER_BY_ID[i], set(ids)) for i in ids if i in core.CAREER_BY_ID]


@app.put("/api/saved/{career_id}")
def save_career(career_id: str):
    if career_id not in core.CAREER_BY_ID:
        raise HTTPException(404, "Career not found")
    with core.db() as conn:
        conn.execute("INSERT OR IGNORE INTO saved (career_id, created_at) VALUES (?, ?)",
                     (career_id, core._now()))
        return {"career_id": career_id, "saved": True, "saved_count": len(core.saved_ids(conn))}


@app.delete("/api/saved/{career_id}")
def unsave_career(career_id: str):
    with core.db() as conn:
        conn.execute("DELETE FROM saved WHERE career_id = ?", (career_id,))
        return {"career_id": career_id, "saved": False, "saved_count": len(core.saved_ids(conn))}


# --------------------------------------------------------------------------- #
# Advice history + AI advisor
# --------------------------------------------------------------------------- #
@app.get("/api/advice")
def list_advice():
    with core.db() as conn:
        return core.list_advice(conn)


@app.delete("/api/advice/{advice_id}")
def delete_advice(advice_id: int):
    with core.db() as conn:
        cur = conn.execute("DELETE FROM advice WHERE id = ?", (advice_id,))
        if cur.rowcount == 0:
            raise HTTPException(404, "Advice not found")
        return {"deleted": advice_id}


@app.post("/api/chat")
def chat(body: ChatIn):
    """Career advisor. Uses Claude when ANTHROPIC_API_KEY is set, otherwise the built-in rules."""
    with core.db() as conn:
        profile = core.get_profile(conn)

    history = [t.model_dump() for t in body.history]
    reply = core.llm_reply(body.message, history, profile)
    source = "ai"
    if reply is None:
        reply, source = core.rule_based_reply(body.message, profile), "rules"

    advice = None
    with core.db() as conn:
        if body.save_advice:
            advice = core.add_advice(conn, reply)
        advice_count = conn.execute("SELECT COUNT(*) FROM advice").fetchone()[0]
    return {"reply": reply, "source": source, "advice": advice, "advice_count": advice_count}


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    page = core.BASE_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"message": "Futurepath AI API is running. Visit /docs. Put index.html next to main.py to serve the UI."}

import os
import re
import json
import datetime as dt
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine, Column, Integer, String, Date
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo


# --- App setup ---------------------------------------------------------------

app = FastAPI(title="Codle Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./codle.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

Base.metadata.create_all(bind=engine)

# --- Config -----------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))

DAILY_GEN_ENABLED = os.getenv("DAILY_GEN_ENABLED", "true").lower() == "true"
DAILY_GEN_TIME = os.getenv("DAILY_GEN_TIME", "06:00")  # HH:MM (24h)
DAILY_GEN_TZ = os.getenv("DAILY_GEN_TZ", "Europe/Paris")
DAILY_GEN_DIFFICULTY = os.getenv("DAILY_GEN_DIFFICULTY", "rotate")


# --- DB dependency -----------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

scheduler: Optional[AsyncIOScheduler] = None

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

    global scheduler
    if DAILY_GEN_ENABLED:
        tz = ZoneInfo(DAILY_GEN_TZ)
        hour, minute = map(int, DAILY_GEN_TIME.split(":"))
        scheduler = AsyncIOScheduler(timezone=tz)

        def _job():
            # Run inside a fresh db session
            with SessionLocal() as db:
                today = dt.datetime.now(tz).date()
                ensure_problem_for_date(today, db)

        # Run every day at the configured time
        scheduler.add_job(_job, CronTrigger(hour=hour, minute=minute, timezone=tz), id="daily_problem")
        scheduler.start()

@app.on_event("shutdown")
def on_shutdown():
    global scheduler
    if scheduler:
        scheduler.shutdown(wait=False)

# --- Schemas ----------------------------------------------------------------

class HealthResponse(BaseModel):
    ok: bool
    model: str
    ollama: str

class Difficulty(str):
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"

class GenerateRequest(BaseModel):
    difficulty: str = Field(..., pattern=r"^(medium|hard|extreme)$", description="Difficulty of problems")

class ProblemPayload(BaseModel):
    id: Optional[int]
    date: Optional[str]
    title: str
    description: str
    starter_code: str
    language: str = "python"
    difficulty: str

class Problem(Base):
    __tablename__ = "problems"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    starter_code = Column(String, nullable=False)
    language = Column(String, nullable=False, default="python")
    difficulty = Column(String, nullable=False)


class GenerateResponse(BaseModel):
    problem: ProblemPayload
    hints: List[str]
    testsuite: List[Dict[str, Any]]

# --- Utilities --------------------------------------------------------------

JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _escape_json_string(s: str) -> str:
    # Escape backslashes, quotes, and newlines to be valid inside JSON strings
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    return s

def _extract_json(text: str) -> Any:
    """Extract and robustly parse JSON payload from model output.

    Accepts fenced JSON code blocks or raw JSON. Attempts to repair common
    model mistakes: comments, trailing commas, smart quotes, and unescaped
    newlines inside the 'starter_code' field.
    """
    # 1) Prefer a ```json ... ``` fenced block if present
    match = JSON_BLOCK.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()

    # 2) Keep only from the first '{' or '[' through the last '}' or ']'
    first_brace = candidate.find("{")
    first_bracket = candidate.find("[")
    first_positions = [p for p in (first_brace, first_bracket) if p != -1]
    first = min(first_positions) if first_positions else -1
    last = max(candidate.rfind("}"), candidate.rfind("]"))
    if first != -1 and last != -1 and last >= first:
        candidate = candidate[first:last+1]

    # 3) Lightweight repairs before JSON parsing
    cleaned = candidate

    # 3a) Normalize smart quotes
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")

    # 3b) Strip JS-style comments
    cleaned = re.sub(r"/\*[\s\S]*?\*/", "", cleaned)              # block comments
    cleaned = re.sub(r"^\s*//.*$", "", cleaned, flags=re.MULTILINE)  # line comments

    # 3c) Remove trailing commas before } or ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    # 3d) Ensure starter_code is a valid JSON string (escape newlines/quotes)
    def fix_starter_code(m: re.Match) -> str:
        prefix = m.group(1)  # includes the leading "starter_code": "
        body   = m.group(2)  # raw (possibly multi-line) code
        return prefix + _escape_json_string(body) + '"'

    cleaned = re.sub(
        r'("starter_code"\s*:\s*")([\s\S]*?)"(?=\s*[,}\n])',
        fix_starter_code,
        cleaned,
        flags=re.DOTALL,
    )

    # 4) Parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # As a last resort, try once more after removing any remaining trailing commas
        attempt = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            # Surface the original for easier debugging
            raise ValueError(f"Could not parse JSON from model output: {e}")

def ollama_generate(prompt: str, model: Optional[str] = None) -> str:
    model = model or OLLAMA_MODEL
    url = f"{OLLAMA_HOST}/api/generate"
    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama not reachable: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Ollama returns { response: "..." }
    return data.get("response", "").strip()


def check_ollama(model: Optional[str] = None) -> Dict[str, str]:
    model = model or OLLAMA_MODEL
    # Try a tiny generate to verify the model exists
    probe = ollama_generate("Respond with exactly the word: OK", model=model)
    ok = "OK" in probe.upper()
    if not ok:
        raise HTTPException(status_code=502, detail=f"Unexpected response from Ollama model '{model}': {probe[:120]}")
    return {"model": model, "ollama": OLLAMA_HOST}

def pick_difficulty_for_date(d: dt.date) -> str:
    """Rotate medium -> hard -> extreme by date."""
    if DAILY_GEN_DIFFICULTY in {"medium", "hard", "extreme"}:
        return DAILY_GEN_DIFFICULTY
    order = ["medium", "hard", "extreme"]
    return order[d.toordinal() % 3]

def ensure_problem_for_date(target_date: dt.date, db: Session) -> Optional[Problem]:
    """If no problem exists for target_date, generate and persist one. Return the row."""
    existing = db.query(Problem).filter(Problem.date == target_date).first()
    if existing:
        return existing

    # Ensure model is up
    check_ollama()

    agent = LeetAgent()
    difficulty = pick_difficulty_for_date(target_date)

    # 1) Problem
    problem_data = agent.create_problem(difficulty)
    # 2) Hints (generated for API response; not stored yet unless you want to extend schema)
    _ = agent.create_hints(problem_data["title"], problem_data["description"])
    # 3) Testsuite (same note as above)
    _ = agent.create_testsuite(problem_data["title"], problem_data["description"])

    row = Problem(
        date=target_date,
        title=problem_data["title"],
        description=problem_data["description"],
        starter_code=problem_data["starter_code"],
        language=problem_data.get("language", "python"),
        difficulty=difficulty,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

# --- Agent prompts ----------------------------------------------------------

PROBLEM_PROMPT = (
    "You are a coding interview problem designer. Create ONE LeetCode-style problem at the requested difficulty.\n"
    "Return ONLY a JSON code block (```json ... ```), with this exact schema and no extra keys:\n"
    "{\n"
    "  \"title\": string,\n"
    "  \"description\": string,\n"
    "  \"starter_code\": string,\n"
    "  \"language\": \"python\"\n"
    "}\n"
    "Rules:\n"
    "- No markdown outside the JSON block.\n"
    "- No comments in the JSON.\n"
    "- No trailing commas.\n"
    "- The starter_code must be a valid Python 3 function or class skeleton (escape newlines as \\n).\n"
    "- No external libraries; keep the signature practical; avoid trivial problems."
)


HINTS_PROMPT = (
    "You are a helpful coding tutor. Based on the following problem, write EXACTLY 3 concise hints, from gentle to more direct.\n"
    "Return ONLY a JSON array of 3 strings."
)

TESTSUITE_PROMPT = (
    "Design a robust unit test suite for the problem below.\n"
    "Output ONLY JSON: an array of test cases. Each test case must be an object with keys: \n"
    "  - \"input\": the function input (single value or array/obj as appropriate)\n"
    "  - \"expected\": the expected output value.\n"
    "Cover edge cases and typical scenarios. The testsuite must be more than 10 tests."
)

# --- Agent orchestration ----------------------------------------------------

class LeetAgent:
    def __init__(self, model: Optional[str] = None):
        self.model = model or OLLAMA_MODEL

    def _ctx(self, title: str, description: str) -> str:
        return f"Problem Title: {title}\n\nDescription (markdown):\n{description}\n"

    def create_problem(self, difficulty: str) -> Dict[str, Any]:
        prompt = (
            f"Difficulty: {difficulty}.\n\n" + PROBLEM_PROMPT
        )
        raw = ollama_generate(prompt, model=self.model)
        data = _extract_json(raw)
        # Sanity fill defaults
        data.setdefault("language", "python")
        if not all(k in data for k in ("title", "description", "starter_code")):
            raise HTTPException(status_code=502, detail="Model did not return the required fields for problem generation")
        return data

    def create_hints(self, title: str, description: str) -> List[str]:
        prompt = self._ctx(title, description) + "\n" + HINTS_PROMPT
        raw = ollama_generate(prompt, model=self.model)
        hints = _extract_json(raw)
        if not (isinstance(hints, list) and len(hints) == 3 and all(isinstance(h, str) for h in hints)):
            raise HTTPException(status_code=502, detail="Model did not return exactly 3 string hints")
        return hints

    def create_testsuite(self, title: str, description: str) -> List[Dict[str, Any]]:
        prompt = self._ctx(title, description) + "\n" + TESTSUITE_PROMPT
        raw = ollama_generate(prompt, model=self.model)
        suite = _extract_json(raw)
        if not (isinstance(suite, list) and all(isinstance(t, dict) and "input" in t and "expected" in t for t in suite)):
            raise HTTPException(status_code=502, detail="Model did not return a valid testsuite array")
        return suite

# --- Routes -----------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def healthcheck():
    info = check_ollama()
    return HealthResponse(ok=True, model=info["model"], ollama=info["ollama"]) 


@app.post("/problems", response_model=GenerateResponse)
def generate_problem(payload: GenerateRequest, db: Session = Depends(get_db)):
    # Validate difficulty explicitly
    difficulty = payload.difficulty.lower()
    if difficulty not in {"medium", "hard", "extreme"}:
        raise HTTPException(status_code=400, detail="difficulty must be one of: medium, hard, extreme")

    # Ensure model is up
    check_ollama()

    agent = LeetAgent()

    # 1) Problem
    problem_data = agent.create_problem(difficulty)

    # 2) Hints
    hints = agent.create_hints(problem_data["title"], problem_data["description"])

    # 3) Testsuite
    testsuite = agent.create_testsuite(problem_data["title"], problem_data["description"])

    # Persist problem (without hints/tests) to DB
    problem_row = Problem(
        date=dt.date.today(),
        title=problem_data["title"],
        description=problem_data["description"],
        starter_code=problem_data["starter_code"],
        language=problem_data.get("language", "python"),
        difficulty=difficulty,
    )
    db.add(problem_row)
    db.commit()
    db.refresh(problem_row)

    return GenerateResponse(
        problem=ProblemPayload(
            id=problem_row.id,
            date=problem_row.date.isoformat(),
            title=problem_row.title,
            description=problem_row.description,
            starter_code=problem_row.starter_code,
            language=problem_row.language,
            difficulty=problem_row.difficulty,
        ),
        hints=hints,
        testsuite=testsuite,
    )


@app.get("/problems/{date}", response_model=ProblemPayload)
def get_problem(date: str, db: Session = Depends(get_db)):
    # Accept ISO date (YYYY-MM-DD) and convert to actual date for DB comparison
    try:
        target = dt.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    row = db.query(Problem).filter(Problem.date == target).first()
    if not row:
        raise HTTPException(status_code=404, detail="Problem not found")
    return ProblemPayload(
        id=row.id,
        date=row.date.isoformat() if row.date else None,
        title=row.title,
        description=row.description,
        starter_code=row.starter_code,
        language=row.language,
        difficulty=row.difficulty,
    )


@app.get("/")
def root():
    return {"name": app.title, "version": app.version}
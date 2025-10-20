import os
import re
import json
import datetime as dt
from rapidfuzz import fuzz
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, text, inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# --- App setup ---------------------------------------------------------------

app = FastAPI(title="Codle Backend", version="1.2.0")

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

# --- Config -----------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "600"))

DAILY_GEN_ENABLED = os.getenv("DAILY_GEN_ENABLED", "true").lower() == "true"
DAILY_GEN_TIME = os.getenv("DAILY_GEN_TIME", "01:00")
DAILY_GEN_TZ = os.getenv("DAILY_GEN_TZ", "Europe/Paris")
DAILY_GEN_DIFFICULTY = os.getenv("DAILY_GEN_DIFFICULTY", "rotate")

VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", OLLAMA_MODEL)
AUTO_APPLY_VERIFIER_FIXES = os.getenv("AUTO_APPLY_VERIFIER_FIXES", "true").lower() == "true"

# --- DB dependency -----------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

scheduler: Optional[AsyncIOScheduler] = None

def _ensure_schema():
    """
    Tiny auto-migration to add the 'hints' column if it doesn't exist yet.
    Safe for SQLite/Postgres; ignored if already present.
    """
    with engine.begin() as conn:
        inspector = inspect(conn)
        if "problems" not in inspector.get_table_names():
            conn.execute(text("""
                CREATE TABLE problems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    starter_code TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'python',
                    difficulty TEXT NOT NULL,
                    hints TEXT NOT NULL DEFAULT '[]'
                )
            """))
            return

        cols = {c["name"] for c in inspector.get_columns("problems")}
        if "hints" not in cols:
            conn.execute(text("ALTER TABLE problems ADD COLUMN hints TEXT NOT NULL DEFAULT '[]'"))

# --- Models / Schemas -------------------------------------------------------

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
    hints: List[str] = []

class Problem(Base):
    __tablename__ = "problems"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    starter_code = Column(String, nullable=False)
    language = Column(String, nullable=False, default="python")
    difficulty = Column(String, nullable=False)
    hints = Column(String, nullable=False, default="[]")

class GenerateResponse(BaseModel):
    problem: ProblemPayload
    hints: List[str]
    testsuite: List[Dict[str, Any]]
    review: Dict[str, Any] = Field(default_factory=dict)

# --- Utilities --------------------------------------------------------------

JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def _strip_json_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"(?m)^\s*//.*?$", "", s)
    return s

def _extract_json(text: str) -> Any:
    print(text)
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=502, detail="Model returned an empty response")

    matches = JSON_BLOCK.findall(text)
    for m in matches:
        candidate = m.strip()
        candidate = _strip_json_comments(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    ANY_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", re.DOTALL)
    for m in ANY_FENCE.findall(text):
        candidate = m.strip()
        candidate = _strip_json_comments(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    try:
        return json.loads(text.strip())
    except Exception:
        pass

    first_brace = text.find("{")
    first_bracket = text.find("[")
    starts = [i for i in (first_brace, first_bracket) if i != -1]
    if starts:
        start = min(starts)
        for end in range(len(text) - 1, start, -1):
            if text[end] in ("}", "]"):
                candidate = text[start : end + 1].strip()
                try:
                    return json.loads(candidate)
                except Exception:
                    continue

    preview = text.strip().replace("\n", " ")
    if len(preview) > 200:
        preview = preview[:200] + "..."
    raise HTTPException(status_code=502, detail=f"Model did not return valid JSON. Preview: {preview}")

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
    return data.get("response", "").strip()

def check_ollama(model: Optional[str] = None) -> Dict[str, str]:
    model = model or OLLAMA_MODEL
    probe = ollama_generate("Respond with exactly the word: OK", model=model)
    if probe.strip() != "OK":
        raise HTTPException(status_code=502, detail="Ollama did not respond correctly to probe")
    return {"model": model, "ollama": OLLAMA_HOST}

def pick_difficulty_for_date(date: dt.date) -> str:
    """
    When DAILY_GEN_DIFFICULTY='rotate', rotate (Mon/Tue/Wed -> medium, Thu/Fri -> hard, Sat/Sun -> extreme)
    else use the configured value (medium|hard|extreme).
    """
    configured = DAILY_GEN_DIFFICULTY.lower().strip()
    if configured in {"medium", "hard", "extreme"}:
        return configured
    wd = date.weekday()
    if wd <= 2:
        return "medium"
    if wd <= 4:
        return "hard"
    return "extreme"

# --- Agent prompts ----------------------------------------------------------

PROBLEM_PROMPT = (
    "You are a coding interview problem designer. Create ONE LeetCode-style problem at the requested difficulty.\n"
    "You must provide 2 examples.\n"
    "Constraints:\n"
    "- Novel problem (not plagiarized).\n"
    "- starter_code must be valid Python 3 and compilable.\n"
    "- Avoid randomness and I/O; function-style problems only."
    "- You must write the description in markdown."
    "- The function must be named solution."
    "Return ONLY a JSON code block (```json ... ```). The JSON must be STRICT (no comments, no trailing commas, no extra text):\n"
    "{\n"
    '  "title": "…",\n'
    '  "description": "…",\n'
    '  "starter_code": "…",\n'
    '  "language": "python"\n'
    "}\n"
)

HINTS_PROMPT = (
    "You are a helpful coding tutor. Based on the following problem, write EXACTLY 3 concise hints, from gentle to more direct.\n"
    "Return ONLY a JSON array of 3 strings."
)

TESTSUITE_PROMPT = (
    "Design a robust unit test suite for the problem below.\n"
    "Output ONLY JSON: an array of test cases. Each test case must be an object with keys:\n"
    '  - "input": the function input (single value or array/obj as appropriate)\n'
    '  - "expected": the expected output encoded as a JSON literal (numbers use "." for decimals, booleans are true/false, null is null)\n'
    '  - "type": the type of the output\n'
    "Cover edge cases and typical scenarios. The testsuite must contain more than 10 tests.\n"
    "Don't generate the same tests twice.\n"
    "Do not translate or localise any values in the JSON; use standard JSON syntax."
)

REVIEW_PROMPT = (
    "You are a rigorous code problem QA reviewer.\n"
    "Given a proposed coding problem (title, description, starter_code, language, difficulty), its 3 hints, and a testsuite,\n"
    "validate STRICTLY against these rules:\n"
    "1) starter_code must be valid Python 3 (compilable skeleton function or class), with no external libraries.\n"
    "2) description must contain at least two worked 'Example' sections.\n"
    "3) hints must be EXACTLY 3 short strings.\n"
    "4) testsuite must be an array of > 10 test cases; each has 'input' and 'expected'.\n"
    "If something is wrong, propose the SMALLEST fixes possible.\n"
    "Return ONLY a JSON object with fields:\n"
    "{\n"
    '  "ok": boolean,\n'
    '  "issues": string[],\n'
    '  "suggestions": {\n'
    '     "title"?: string,\n'
    '     "description"?: string,\n'
    '     "starter_code"?: string,\n'
    '     "hints"?: string[3],\n'
    '     "testsuite"?: {"replace": boolean, "items": array}\n'
    "  }\n"
    "}\n"
    "Rules: output must be valid JSON; keep suggestions minimal; do not invent new APIs."
)

# --- Agent orchestration ----------------------------------------------------

class LeetAgent:
    def __init__(self, db: Session = Depends(get_db), model: Optional[str] = None):
        self.db = db
        self.model = model or OLLAMA_MODEL

    def _ctx(self, title: str, description: str) -> str:
        return f"Problem Title: {title}\n\nDescription (markdown):\n{description}\n"

    def create_problem(self, difficulty: str) -> Dict[str, Any]:
        all_rows = self.db.query(Problem.id, Problem.title, Problem.description).all()
        threshold = 80
        max_attempts = 10
        for _ in range(max_attempts):
            prompt = f"Difficulty: {difficulty}.\n\n" + PROBLEM_PROMPT
            raw = ollama_generate(prompt, model=self.model)
            data = _extract_json(raw)
            data.setdefault("language", "python")
            if not all(k in data for k in ("title", "description", "starter_code")):
                continue

            needle = f"{data['title']} {data['description']}"
            max_sim = 0
            for row in all_rows:
                text_b = f"{row.title} {row.description}"
                sim = fuzz.token_sort_ratio(needle, text_b)
                if sim > max_sim:
                    max_sim = sim
                if max_sim >= threshold:
                    break
            if max_sim < threshold:
                return data

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
        # Normalize the suite so that 'expected' values are encoded as JSON literals.
        normalized = []
        for t in suite:
            try:
                # Use json.dumps to serialise the expected value into a JSON literal. This
                # ensures that numbers use '.' for decimals, booleans are true/false and
                # lists/objects are rendered correctly. Strings will be quoted, which is
                # acceptable because the client parses them with JSON.parse later.
                norm_expected = json.dumps(t["expected"], ensure_ascii=False)
            except Exception:
                # Fallback: coerce to string and dump again.
                norm_expected = json.dumps(str(t.get("expected")), ensure_ascii=False)
            # Create a shallow copy to avoid mutating original objects coming from the model.
            tc = dict(t)
            tc["expected"] = norm_expected
            normalized.append(tc)
        return normalized

class VerifierAgent:
    """Second agent that reviews and suggests minimal corrections."""
    def __init__(self, model: Optional[str] = None):
        self.model = model or VERIFIER_MODEL

    @staticmethod
    def _local_checks(problem_data: Dict[str, Any], hints: List[str], suite: List[Dict[str, Any]]) -> List[str]:
        issues: List[str] = []
        code = problem_data.get("starter_code", "")
        try:
            compiled = code.replace("\\n", "\n")
            compile(compiled, "<starter_code>", "exec")
        except Exception as e:
            issues.append(f"starter_code does not compile: {type(e).__name__}: {e}")
        desc = problem_data.get("description", "")
        if len(re.findall(r"\bExample\b", desc, flags=re.IGNORECASE)) < 2:
            issues.append("description appears to have fewer than 2 'Example' sections")
        if not (isinstance(hints, list) and len(hints) == 3 and all(isinstance(h, str) for h in hints)):
            issues.append("hints are not exactly 3 strings")
        if not (isinstance(suite, list) and len(suite) > 10 and all(isinstance(t, dict) and "input" in t and "expected" in t for t in suite)):
            issues.append("testsuite invalid or has <= 10 tests")
        return issues

    def review(self, problem_data: Dict[str, Any], hints: List[str], suite: List[Dict[str, Any]], difficulty: str) -> Dict[str, Any]:
        local_issues = self._local_checks(problem_data, hints, suite)
        payload = {
            "title": problem_data.get("title"),
            "description": problem_data.get("description"),
            "starter_code": problem_data.get("starter_code"),
            "language": problem_data.get("language", "python"),
            "difficulty": difficulty,
            "hints": hints,
            "testsuite": suite,
        }
        prompt = "Input JSON:\n" + json.dumps(payload, ensure_ascii=False) + "\n\n" + REVIEW_PROMPT
        raw = ollama_generate(prompt, model=self.model)
        data = _extract_json(raw)
        if not isinstance(data, dict) or "ok" not in data or "issues" not in data:
            raise HTTPException(status_code=502, detail="Verifier did not return the required fields")
        data.setdefault("issues", [])
        if local_issues:
            merged = []
            seen = set()
            for it in [*local_issues, *data["issues"]]:
                if it not in seen:
                    merged.append(it)
                    seen.add(it)
            data["issues"] = merged
        data.setdefault("suggestions", {})
        return data

def _apply_suggestions(problem_data: Dict[str, Any], hints: List[str], suite: List[Dict[str, Any]], suggestions: Dict[str, Any]):
    updated_problem = dict(problem_data)
    updated_hints = list(hints)
    updated_suite = list(suite)
    if not isinstance(suggestions, dict):
        return updated_problem, updated_hints, updated_suite
    for k in ("title", "description", "starter_code"):
        if k in suggestions and isinstance(suggestions[k], str) and suggestions[k].strip():
            updated_problem[k] = suggestions[k]
    if "hints" in suggestions and isinstance(suggestions["hints"], list) and len(suggestions["hints"]) == 3 and all(isinstance(h, str) for h in suggestions["hints"]):
        updated_hints = suggestions["hints"]
    if "testsuite" in suggestions and isinstance(suggestions["testsuite"], dict):
        ts = suggestions["testsuite"]
        if ts.get("replace") and isinstance(ts.get("items"), list):
            # Normalise expected values in suggested testsuite. Use json.dumps to ensure JSON
            # primitives are encoded canonically. See _normalize_testsuite in LeetAgent.
            new_suite = []
            for t in ts["items"]:
                try:
                    norm_expected = json.dumps(t["expected"], ensure_ascii=False)
                except Exception:
                    norm_expected = json.dumps(str(t.get("expected")), ensure_ascii=False)
                tc = dict(t)
                tc["expected"] = norm_expected
                new_suite.append(tc)
            updated_suite = new_suite
    return updated_problem, updated_hints, updated_suite

# --- Daily generation --------------------------------------------------------

def ensure_problem_for_date(target_date: dt.date, db: Session) -> Optional[Problem]:
    existing = db.query(Problem).filter(Problem.date == target_date).first()
    if existing:
        return existing

    check_ollama()

    agent = LeetAgent(db)
    difficulty = pick_difficulty_for_date(target_date)

    problem_data = agent.create_problem(difficulty)
    hints = agent.create_hints(problem_data["title"], problem_data["description"])
    testsuite = agent.create_testsuite(problem_data["title"], problem_data["description"])

    verifier = VerifierAgent()
    review = verifier.review(problem_data, hints, testsuite, difficulty)
    if AUTO_APPLY_VERIFIER_FIXES and isinstance(review.get("suggestions"), dict):
        problem_data, hints, testsuite = _apply_suggestions(problem_data, hints, testsuite, review["suggestions"])

    row = Problem(
        date=target_date,
        title=problem_data["title"],
        description=problem_data["description"],
        starter_code=problem_data["starter_code"],
        language=problem_data.get("language", "python"),
        difficulty=difficulty,
        hints=json.dumps(hints),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

def _schedule_daily():
    global scheduler
    if not DAILY_GEN_ENABLED:
        return
    if scheduler is None:
        scheduler = AsyncIOScheduler(timezone=DAILY_GEN_TZ)
        scheduler.start()

    try:
        hh, mm = map(int, DAILY_GEN_TIME.split(":"))
        hh = max(0, min(23, hh))
        mm = max(0, min(59, mm))
    except Exception:
        hh, mm = 6, 0

    def job():
        with SessionLocal() as db:
            ensure_problem_for_date(dt.date.today(), db)

    for job_ in scheduler.get_jobs():
        job_.remove()

    scheduler.add_job(job, "cron", hour=hh, minute=mm, id="daily-problem")

# --- Routes -----------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    _ensure_schema()
    _schedule_daily()

@app.get("/check-health", response_model=HealthResponse)
def check_health():
    try:
        info = check_ollama()
        return HealthResponse(ok=True, model=info["model"], ollama=info["ollama"])
    except HTTPException as e:
        raise e

@app.post("/problems", response_model=GenerateResponse)
def generate_problem(payload: GenerateRequest, db: Session = Depends(get_db)):
    difficulty = payload.difficulty.lower()
    if difficulty not in {"medium", "hard", "extreme"}:
        raise HTTPException(status_code=400, detail="difficulty must be one of: medium, hard, extreme")

    check_ollama()
    agent = LeetAgent(db)

    problem_data = agent.create_problem(difficulty)
    hints = agent.create_hints(problem_data["title"], problem_data["description"])
    testsuite = agent.create_testsuite(problem_data["title"], problem_data["description"])

    verifier = VerifierAgent()
    review = verifier.review(problem_data, hints, testsuite, difficulty)
    if AUTO_APPLY_VERIFIER_FIXES and isinstance(review.get("suggestions"), dict):
        problem_data, hints, testsuite = _apply_suggestions(problem_data, hints, testsuite, review["suggestions"])

    problem_row = Problem(
        date=dt.date.today(),
        title=problem_data["title"],
        description=problem_data["description"],
        starter_code=problem_data["starter_code"],
        language=problem_data.get("language", "python"),
        difficulty=difficulty,
        hints=json.dumps(hints),
    )
    db.add(problem_row)
    db.commit()
    db.refresh(problem_row)

    return GenerateResponse(
        problem=ProblemPayload(
            id=problem_row.id,
            date=problem_row.date.isoformat() if problem_row.date else None,
            title=problem_row.title,
            description=problem_row.description,
            starter_code=problem_row.starter_code,
            language=problem_row.language,
            difficulty=problem_row.difficulty,
            hints=hints,
        ),
        hints=hints,
        testsuite=testsuite,
        review=review,
    )

@app.get("/problems/{date}", response_model=ProblemPayload)
def get_problem_by_date(date: str, db: Session = Depends(get_db)):
    try:
        target = dt.datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")

    row = db.query(Problem).filter(Problem.date == target).first()
    if not row:
        raise HTTPException(status_code=404, detail="Problem not found")
    try:
        hints = json.loads(row.hints) if row.hints else []
    except json.JSONDecodeError:
        hints = []
    return ProblemPayload(
        id=row.id,
        date=row.date.isoformat() if row.date else None,
        title=row.title,
        description=row.description,
        starter_code=row.starter_code,
        language=row.language,
        difficulty=row.difficulty,
        hints=hints,
    )

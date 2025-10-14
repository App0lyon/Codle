# dump_db.py
import os
import argparse
import datetime as dt
from typing import List

from sqlalchemy import create_engine, select
from sqlalchemy.orm import declarative_base, Session, Mapped, mapped_column

# Même valeur par défaut que ton serveur FastAPI
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./codle.db")  # :contentReference[oaicite:1]{index=1}
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
Base = declarative_base()

class Problem(Base):
    __tablename__ = "problems"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    date: Mapped[dt.date | None]
    title: Mapped[str]
    description: Mapped[str]
    starter_code: Mapped[str]
    language: Mapped[str]
    difficulty: Mapped[str]
    # champs/nom de table alignés sur server.py :contentReference[oaicite:2]{index=2}

def list_all(limit: int = 50) -> List[Problem]:
    with Session(engine) as s:
        stmt = select(Problem).order_by(Problem.id.desc()).limit(limit)
        return list(s.scalars(stmt))

def get_by_id(pid: int) -> Problem | None:
    with Session(engine) as s:
        return s.get(Problem, pid)

def list_by_date(date_str: str) -> List[Problem]:
    date = dt.date.fromisoformat(date_str)
    with Session(engine) as s:
        stmt = select(Problem).where(Problem.date == date).order_by(Problem.id.desc())
        return list(s.scalars(stmt))

def search_title(q: str, limit: int = 50) -> List[Problem]:
    with Session(engine) as s:
        stmt = select(Problem).where(Problem.title.ilike(f"%{q}%")).order_by(Problem.id.desc()).limit(limit)
        return list(s.scalars(stmt))

def pretty(p: Problem) -> str:
    d = p.date.isoformat() if p.date else "—"
    return f"[{p.id}] ({d}) {p.difficulty.upper()} — {p.title} [{p.language}]"

def main():
    ap = argparse.ArgumentParser(description="Lire la DB codle")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_list = sub.add_parser("list", help="Lister les derniers problèmes")
    ap_list.add_argument("-n", "--limit", type=int, default=20)

    ap_get = sub.add_parser("get", help="Afficher un problème par id")
    ap_get.add_argument("id", type=int)

    ap_date = sub.add_parser("date", help="Lister par date (YYYY-MM-DD)")
    ap_date.add_argument("date")

    ap_search = sub.add_parser("search", help="Chercher dans le titre")
    ap_search.add_argument("query")
    ap_search.add_argument("-n", "--limit", type=int, default=20)

    args = ap.parse_args()

    if args.cmd == "list":
        for p in list_all(args.limit):
            print(pretty(p))
    elif args.cmd == "get":
        p = get_by_id(args.id)
        if not p:
            print("Not found")
            return
        print(pretty(p))
        print("\nDescription:\n", p.description)
        print("\nStarter code:\n", p.starter_code)
    elif args.cmd == "date":
        for p in list_by_date(args.date):
            print(pretty(p))
    elif args.cmd == "search":
        for p in search_title(args.query, args.limit):
            print(pretty(p))

if __name__ == "__main__":
    # Crée les métadonnées si besoin (no-op si la table existe déjà)
    Base.metadata.create_all(bind=engine)
    main()

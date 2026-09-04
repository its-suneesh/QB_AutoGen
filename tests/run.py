"""Runs the test suites and reports which passed.

    python tests/run.py            everything it can
    python tests/run.py --quick    only the suites that need no database

Each suite is a standalone script that exits non-zero on failure, so this only
has to run them and collect the exit codes. That keeps them runnable one at a
time while debugging - `python tests/test_books.py` - which is how they are
usually read.

Two of them need Postgres with pgvector. Rather than fail when there is none,
they are SKIPPED with the reason said out loud: a run that silently tested half
of what you thought is worse than one that tested less and told you.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (file, needs a database)
SUITES = [
    ("test_schema.py", False),      # LaTeX-in-JSON repair
    ("test_retrieval.py", False),   # topic splitting, page windows, citations
    ("test_search.py", False),      # search against a stand-in for pgvector
    ("test_missing.py", False),     # every way a book can fail to be retrievable
    ("test_books.py", True),        # book lifecycle against a real database
    ("test_migrate.py", True),      # schema upgrades must not lose data
]


def database_reachable() -> tuple[bool, str]:
    """Whether the configured store can be reached, and why not if it cannot."""
    url = os.environ.get("RAG_DATABASE_URL", "")
    if not url:
        return False, "RAG_DATABASE_URL is not set"

    try:
        import psycopg
    except ImportError:
        return False, "psycopg is not installed"

    try:
        with psycopg.connect(url, connect_timeout=5) as conn:
            row = conn.execute(
                "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if not row[0]:
                return False, "the vector extension is not enabled on that database"
        return True, ""
    except Exception as exc:
        return False, str(exc).strip().splitlines()[0][:70]


def main() -> int:
    quick = "--quick" in sys.argv

    if quick:
        has_db, why = False, "--quick"
    else:
        has_db, why = database_reachable()

    env = dict(os.environ, PYTHONPATH=ROOT, RAG_ENABLED="true")

    print("QB AutoGen test suites\n")
    if not has_db:
        print(f"  database suites will be SKIPPED: {why}\n")

    failed, skipped, passed = [], [], []
    for name, needs_db in SUITES:
        if needs_db and not has_db:
            print(f"  {name:<20} SKIPPED")
            skipped.append(name)
            continue

        started = time.time()
        result = subprocess.run(
            [sys.executable, os.path.join(HERE, name)],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        took = time.time() - started

        if result.returncode == 0:
            print(f"  {name:<20} ok        {took:5.1f}s")
            passed.append(name)
        else:
            print(f"  {name:<20} FAILED    {took:5.1f}s")
            failed.append(name)
            for line in (result.stdout + result.stderr).splitlines():
                if "FAIL" in line or "Error" in line or "error" in line:
                    print(f"      {line.strip()[:110]}")

    print(f"\n  {len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if skipped:
        print("  run the skipped ones with a pgvector database:")
        print("    docker run -d --name qbrag -e POSTGRES_USER=qbrag "
              "-e POSTGRES_PASSWORD=qbrag -e POSTGRES_DB=qbrag "
              "-p 55432:5432 pgvector/pgvector:pg16")
        print("    RAG_DATABASE_URL=postgresql://qbrag:qbrag@localhost:55432/qbrag "
              "python tests/run.py")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

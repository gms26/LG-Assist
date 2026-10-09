"""Shared fixtures for DB-backed tests.

These tests need a REAL Postgres with the pgvector extension. They are
skipped when the test database is unreachable.

Run them with the stack up (or any reachable Postgres):
    LGA_TEST_DATABASE_URL=postgresql+psycopg://lga:lga@localhost:5432/lga_test \\
        pytest tests/test_api_db.py

SAFETY: fixtures drop all tables first, so the database name MUST contain
'test' — the fixture refuses anything else.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

TEST_DB_URL = os.environ.get(
    "LGA_TEST_DATABASE_URL",
    "postgresql+psycopg://lga:lga@localhost:5432/lga_test",
)


@pytest.fixture(scope="session")
def test_engine():
    db_name = TEST_DB_URL.rsplit("/", 1)[-1].split("?")[0]
    if "test" not in db_name:
        pytest.skip("refusing to drop tables: test DB name must contain 'test'")
    engine = create_engine(TEST_DB_URL)
    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:
        pytest.skip(f"test database unreachable: {exc}")
    from app.models import Base

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db(test_engine):
    """Isolated session: rolled back after each test."""
    conn = test_engine.connect()
    trans = conn.begin()
    session = sessionmaker(bind=conn)()
    yield session
    session.close()
    trans.rollback()
    conn.close()

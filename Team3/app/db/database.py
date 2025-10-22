# db/database.py
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

def make_sqlite_url(db_path: str) -> str:
    db_path = os.path.abspath(db_path)
    return f"sqlite:///{db_path}"

def create_engine_and_session(db_path: str):
    url = make_sqlite_url(db_path)
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()

    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine, SessionLocal

def init_db(engine):
    from db.models import Base
    Base.metadata.create_all(engine)

# utils/db.py
import os
import sqlalchemy as sa
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Build a correct SQL-Alchemy 2.x URL for SQL-Server trusted auth
# ------------------------------------------------------------------
def get_engine() -> sa.Engine:
    host = os.getenv("SQL_HOST")
    db   = os.getenv("SQL_DB")
    driver = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server").replace(" ", "+")

    # Trusted connection ⇒ no UID/PWD in URL
    url = sa.URL.create(
        drivername="mssql+pyodbc",
        username="",           # empty
        password="",           # empty
        host=host,
        database=db,
        query={
            "driver": driver,
            "trusted_connection": "yes",
            "charset": "UTF-8",
        },
    )
    return create_engine(url, pool_pre_ping=True, fast_executemany=True)

# ------------------------------------------------------------------
# Thin wrapper: execute any SELECT and return DataFrame
# ------------------------------------------------------------------
def fetch_df(query: str, params: dict | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.begin() as conn:          # auto-commit/rollback
        return pd.read_sql_query(text(query), conn, params=params)
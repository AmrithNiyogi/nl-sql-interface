import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Optional: Read DB URL from environment or set directly here
DB_URL = os.getenv("POSTGRES_DB_URL", "postgresql://postgres:postgres@localhost:5432/chinook")

def get_engine() -> Engine:
    """
    Create and return an SQLAlchemy engine for connecting to the PostgreSQL database.

    Returns:
        Engine: SQLAlchemy engine
    """
    return create_engine(DB_URL, pool_size=10, max_overflow=20)

def execute_query(query: str, return_df: bool = True):
    """
    Executes a raw SQL query and returns results as a list or DataFrame.

    Args:
        query (str): SQL query string
        return_df (bool): Whether to return result as pandas DataFrame

    Returns:
        (list | pd.DataFrame): Query result
    """
    engine = get_engine()
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            if result.returns_rows:
                if return_df:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return df
                else:
                    return result.fetchall()
            else:
                # Returning an empty DataFrame when no rows are returned
                return pd.DataFrame()
    except SQLAlchemyError as e:
        logging.error(f"Error executing query: {str(e)}")
        return f"Error executing query: {str(e)}"

def get_all_table_names():
    """
    Get a list of all table names in the current database schema.

    Returns:
        list: List of table names
    """
    engine = get_engine()
    try:
        with engine.connect() as connection:
            result = connection.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """))
            return [row[0] for row in result]
    except SQLAlchemyError as e:
        logging.error(f"Error fetching table names: {str(e)}")
        return f"Error fetching table names: {str(e)}"

def get_table_schema(table_name: str):
    """
    Get the schema of a specific table, including column names and data types.

    Args:
        table_name (str): Name of the table

    Returns:
        list: List of (column_name, data_type) tuples
    """
    engine = get_engine()
    try:
        with engine.connect() as connection:
            result = connection.execute(text(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = :table
            """), {"table": table_name})
            return [(row[0], row[1]) for row in result]
    except SQLAlchemyError as e:
        logging.error(f"Error fetching schema for {table_name}: {str(e)}")
        return f"Error fetching schema for {table_name}: {str(e)}"

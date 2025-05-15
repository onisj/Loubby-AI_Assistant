import psycopg2
from psycopg2 import Error
import os
from dotenv import load_dotenv

load_dotenv()

# Connection parameters for the default 'postgres' database
default_conn_params = {
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("POSTGRES_PASSWORD", "isinmiboie16#_"),
    "host": "localhost",
    "port": "5432"
}

# Target database URI from .env
target_uri = os.getenv("POSTGRES_URI", "postgresql://postgres@localhost:5432/loubby_db")
target_dbname = "loubby_db"

try:
    # Connect to the default 'postgres' database to create the new database
    conn = psycopg2.connect(**default_conn_params)
    conn.autocommit = True  # Needed for database creation
    cur = conn.cursor()

    # Check if the database already exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_dbname,))
    exists = cur.fetchone()

    if not exists:
        # Create the database
        cur.execute(f"CREATE DATABASE {target_dbname}")
        print(f"Database '{target_dbname}' created successfully.")
    else:
        print(f"Database '{target_dbname}' already exists.")

    cur.close()
    conn.close()

    # Connect to the new database to set up schema
    conn = psycopg2.connect(target_uri)
    conn.autocommit = True
    cur = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("pgvector extension enabled.")

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs_collection (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            section VARCHAR(255),
            embedding VECTOR(1024)
        );
    """)
    print("Table 'docs_collection' created.")

    # Create vector search index
    cur.execute("""
        CREATE INDEX IF NOT EXISTS docs_embedding_idx 
        ON docs_collection 
        USING hnsw (embedding vector_cosine_ops);
    """)
    print("Vector search index created.")

    cur.close()
    conn.close()
    print("Database setup completed successfully!")

except Error as e:
    print(f"Error setting up database: {e}")
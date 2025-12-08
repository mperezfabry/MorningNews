import sqlite3
import os

# PATH CONFIGURATION
# I'm setting up dynamic paths here so this script works regardless of where it's run from.
# This ensures it always finds 'morningnews.db' in the 'data' folder and 'schema.sql' in the root.
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'morningnews.db')
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'schema.sql')

"""
The update_db function applies the latest schema changes to the SQLite database.

I added this utility script so we don't have to manually run SQL commands
or delete the database every time we add a new table (like 'bookmarks').
"""


# Sanity check to ensure the database exists before trying to update it.
def update_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # This reads the full SQL schema file.
    # This allows us to maintain a single source of truth for our database structure in 'schema.sql'.
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)

# executescript() runs the entire SQL file at once.
# This is perfect for applying new 'CREATE TABLE IF NOT EXISTS' statements without breaking existing data.
    try:
        conn.executescript(schema_sql)
        print("Database schema updated successfully.")

# Catch any SQL errors (like syntax issues in schema.sql) and print them clearly.
    except Exception as e:
        print(f"Error updating schema: {e}")

# Always close the connection to prevent database locks, even if an error occurs.
    finally:
        conn.close()


if __name__ == "__main__":
    update_db()

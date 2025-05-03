import sqlite3

# Connect to SQLite (Creates 'data.db' in the current folder if not exists)
conn = sqlite3.connect("data.db")
cursor = conn.cursor()

# Commit and close
conn.commit()
conn.close()

print("SQLite database created successfully!")

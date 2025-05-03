import sqlite3
import pandas as pd

def init_amazon_database():
    try:
        # Connect to Amazon Prime database
        conn = sqlite3.connect('amazon_prime.db')
        cursor = conn.cursor()
        
        # Create amazon_prime_titles table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS amazon_prime_titles (
                show_id TEXT PRIMARY KEY,
                type TEXT,
                title TEXT,
                director TEXT,
                cast TEXT,
                country TEXT,
                date_added TEXT,
                release_year INTEGER,
                rating TEXT,
                duration TEXT,
                listed_in TEXT,
                description TEXT
            )
        ''')
        
        # Check if table is empty
        cursor.execute('SELECT COUNT(*) FROM amazon_prime_titles')
        if cursor.fetchone()[0] == 0:
            # Import Amazon Prime data from CSV
            print("Importing Amazon Prime data from CSV...")
            df = pd.read_csv('amazon_prime_titles.csv')
            df.to_sql('amazon_prime_titles', conn, if_exists='append', index=False)
            print("Amazon Prime data imported successfully!")
        else:
            print("Amazon Prime data already exists in the database.")
        
        conn.commit()
        conn.close()
        print("Amazon Prime database initialization completed!")
        return True
    except Exception as e:
        print(f"Error initializing Amazon Prime database: {str(e)}")
        return False

if __name__ == "__main__":
    init_amazon_database() 
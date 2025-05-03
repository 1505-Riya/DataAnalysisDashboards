import sqlite3
import pandas as pd

def init_database():
    try:
        # Connect to database
        conn = sqlite3.connect('ecommerce.db')
        cursor = conn.cursor()
        
        # Create netflix_titles table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS netflix_titles (
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
        
        # Check if netflix_titles table is empty
        cursor.execute('SELECT COUNT(*) FROM netflix_titles')
        if cursor.fetchone()[0] == 0:
            # Import Netflix data from CSV
            print("Importing Netflix data from CSV...")
            df = pd.read_csv('netflix_titles.csv')
            df.to_sql('netflix_titles', conn, if_exists='append', index=False)
            print("Netflix data imported successfully!")
        else:
            print("Netflix data already exists in the database.")
            
        # Check if amazon_prime_titles table is empty
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
        print("Database initialization completed successfully!")
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    init_database() 
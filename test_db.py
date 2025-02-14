import os
import psycopg2
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

def test_connection():
    try:
        password = quote_plus(os.getenv('PGPASSWORD', ''))
        conn = psycopg2.connect(
            host=os.getenv('PGHOST'),
            database=os.getenv('PGDATABASE'),
            user=os.getenv('PGUSER'),
            password=password,
            sslmode='disable'
        )
        print("Connection successful!")
        
        # Test a simple query
        cur = conn.cursor()
        cur.execute('SELECT version();')
        version = cur.fetchone()
        print(f"PostgreSQL version: {version[0]}")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        print("\nTrying to get more connection details...")
        try:
            # Try to print more connection details for debugging
            print(f"Host: {os.getenv('PGHOST')}")
            print(f"Database: {os.getenv('PGDATABASE')}")
            print(f"User: {os.getenv('PGUSER')}")
            print(f"SSL Mode: disable")
        except Exception as debug_e:
            print(f"Error printing debug info: {debug_e}")

if __name__ == "__main__":
    test_connection() 
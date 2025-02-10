import psycopg2
from config import DATABASE_URL
import sys
import os

def get_db_params(database_url):
    # Remove postgresql+asyncpg:// from the start
    clean_url = database_url.replace('postgresql+asyncpg://', '')
    # Split user:pass@host:port/dbname
    auth, rest = clean_url.split('@')
    user, password = auth.split(':')
    host_port, dbname = rest.split('/')
    
    return {
        'dbname': dbname,
        'user': user,
        'password': password,
        'host': host_port.split(':')[0]
    }

if __name__ == "__main__":
    db_params = get_db_params(DATABASE_URL)
    
    # Use psql if available, otherwise fallback to python shell
    if os.system('which psql >/dev/null 2>&1') == 0:
        os.system(f"psql -h {db_params['host']} -U {db_params['user']} {db_params['dbname']}")
    else:
        conn = psycopg2.connect(**db_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("Connected to database. Type SQL commands or 'exit' to quit.")
        while True:
            try:
                command = input("sql> ")
                if command.lower() in ('exit', 'quit', '\q'):
                    break
                if command.strip():
                    cursor.execute(command)
                    if cursor.description:
                        results = cursor.fetchall()
                        for row in results:
                            print(row)
            except Exception as e:
                print(f"Error: {e}") 
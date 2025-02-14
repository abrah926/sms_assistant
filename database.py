from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"Connecting to Supabase at: {SUPABASE_URL}")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_connection():
    try:
        print("Testing connection...")
        
        # First try to select from training_examples
        print("Checking for existing tables...")
        response = supabase.table('training_examples').select("*").limit(1).execute()
        print("Found training_examples table")
        
        # If that worked, try inserting a test record
        test_data = {
            'input_text': 'test connection',
            'output_text': 'test successful',
            'created_at': 'now()'
        }
        
        print("Inserting test record...")
        response = supabase.table('training_examples').insert(test_data).execute()
        
        print(f"Response: {response.data}")
        print("✅ Connected to Supabase!")
        return True
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        print(f"Type of error: {type(e)}")
        return False

if __name__ == "__main__":
    print("Starting connection test...")
    test_connection() 
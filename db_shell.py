from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Message
from config import DATABASE_URL

# Convert async URL to sync URL
SYNC_DB_URL = DATABASE_URL.replace('+asyncpg', '')

def check_messages():
    try:
        print(f"Connecting to database: {SYNC_DB_URL}")
        engine = create_engine(SYNC_DB_URL)
        Session = sessionmaker(engine)
        
        with Session() as session:
            print("\nFetching recent messages...")
            messages = session.query(Message).order_by(Message.timestamp.desc()).limit(5).all()
            
            if not messages:
                print("No messages found in database")
                return
            
            print("\nMost recent messages:")
            print("="*50)
            for msg in messages:
                print(f"Direction: {msg.direction}")
                print(f"Phone: {msg.phone}")
                print(f"Content: {msg.content}")
                print(f"Timestamp: {msg.timestamp}")
                print("-"*50)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    check_messages() 
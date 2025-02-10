from matrix_client.client import MatrixClient
from config import MATRIX_HOMESERVER_URL, MATRIX_ACCESS_TOKEN

def test_connection():
    try:
        client = MatrixClient(MATRIX_HOMESERVER_URL)
        client.login_with_token(MATRIX_ACCESS_TOKEN)
        print("Matrix connection successful!")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_connection() 
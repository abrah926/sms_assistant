import psycopg2

# ✅ Explicitly define Azure PostgreSQL connection parameters
DB_NAME = "postgres"  # Change if using a different DB name
DB_USER = "abrah926"
DB_PASSWORD = "Micasa1758"
DB_HOST = "sms1758.postgres.database.azure.com"
DB_PORT = "5432"

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        sslmode="require",
    )
    print("✅ Successfully connected to Azure PostgreSQL!")
    conn.close()
except psycopg2.OperationalError as e:
    print(f"❌ PostgreSQL Connection Error: {e}")
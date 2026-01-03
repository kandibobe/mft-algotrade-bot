import socket
import sys
from sqlalchemy import create_engine, text

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3) # Timeout after 3 seconds
    result = sock.connect_ex((host, port))
    sock.close()
    if result == 0:
        print(f"✅ Port {port} is OPEN")
        return True
    else:
        print(f"❌ Port {port} is CLOSED. Please run 'docker ps'")
        return False

def check_db_connection():
    db_url = "postgresql://user:password@127.0.0.1:5433/trading_analytics"
    print(f"Attempting to connect to DB: {db_url} ...")
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful! (SELECT 1 returned)")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    print("Checking infrastructure...")
    host = "127.0.0.1"
    port = 5433
    
    is_open = check_port(host, port)
    
    if is_open:
        check_db_connection()
    else:
        print("Skipping DB connection check because port is closed.")

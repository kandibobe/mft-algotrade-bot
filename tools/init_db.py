
import sys
import os

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.db_manager import DatabaseManager, Base
# Importing the models is crucial so they are registered with Base.metadata
from src.database.models import TradeRecord, SignalRecord, ExecutionRecord, SystemEvent

def main():
    print("Initializing database...")
    try:
        engine = DatabaseManager.get_engine()
        
        # This will create all tables that are registered with Base.metadata
        Base.metadata.create_all(bind=engine)
        
        print("✅ Successfully created database tables: trades, signals, executions, system_events")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

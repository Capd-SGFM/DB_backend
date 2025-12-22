
import sys
import os

# Set Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from db_module.connect_sqlalchemy_engine import SyncSessionLocal

def ensure_vip_state():
    print("[INIT] Ensuring VIP_BACKTESTING state (ID=7)...")
    with SyncSessionLocal() as session:
        # Check if ID=7 exists
        res = session.execute(text("SELECT id FROM trading_data.pipeline_state WHERE id=7")).scalar()
        if not res:
            print("[INIT] ID=7 missing. Inserting...")
            session.execute(text("""
                INSERT INTO trading_data.pipeline_state (id, is_active, updated_at)
                VALUES (7, false, now())
            """))
            session.commit()
            print("[INIT] Inserted ID=7 (VIP_BACKTESTING).")
        else:
            print("[INIT] ID=7 already exists.")

if __name__ == "__main__":
    ensure_vip_state()

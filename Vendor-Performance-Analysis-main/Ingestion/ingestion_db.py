import pandas as pd
import os
import time
import logging
from sqlalchemy import create_engine

# ---------- Setup logging ----------
LOG_DIR = "logs"
LOG_FILE = "ingestion_db.log"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("ingestion_logger")
logger.setLevel(logging.INFO)

# Remove old handlers if script was run multiple times
if logger.hasHandlers():
    logger.handlers.clear()

# File handler with formatting
file_handler = logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE), mode="a", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# ---------- DB Engine ----------
engine = create_engine("sqlite:///inventory.db")

# ---------- Ingest Function ----------
def ingest_db(df, table_name, engine):
    try:
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        logger.info(f"✅ Ingested table: {table_name}")
    except Exception as e:
        logger.error(f"❌ Failed to ingest table {table_name}: {e}")

# ---------- Load and Ingest All CSVs ----------
def load_raw_data():
    start_time = time.time()
    logger.info("🚀 Starting CSV ingestion")

    try:
        files = os.listdir("Inventory Analysis Dataset")
        csv_files = [f for f in files if f.endswith(".csv")]

        for file in csv_files:
            filepath = os.path.join("Inventory Analysis Dataset", file)
            try:
                df = pd.read_csv(filepath)
                table_name = file[:-4]  # remove .csv
                ingest_db(df, table_name, engine)
            except Exception as e:
                logger.error(f"❌ Error reading file {file}: {e}")
    except Exception as e:
        logger.critical(f"❌ Failed to list files in directory: {e}")

    total_time = (time.time() - start_time) / 60
    logger.info(f"🏁 Ingestion complete in {total_time:.2f} minutes")

# ---------- Run if Main ----------
if __name__ == "__main__":
    load_raw_data()

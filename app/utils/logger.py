# app/utils/logger.py
import logging
import os
from pythonjsonlogger import jsonlogger

LOG_DIR = os.getenv("APP_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("kidney_app")
logger.setLevel(logging.INFO)

# File handler (JSON)
fh = logging.FileHandler(LOG_FILE)
fh.setFormatter(jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

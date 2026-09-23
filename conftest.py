import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Local runs read keys from .env; CI sets real environment variables, which win.
load_dotenv(ROOT / ".env")

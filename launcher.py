import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.launcher import *

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

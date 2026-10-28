from pathlib import Path
import sys

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.api import create_app


if __name__ == "__main__":
    app = create_app(ROOT / "config.yaml")
    uvicorn.run(app, host="0.0.0.0", port=8000)

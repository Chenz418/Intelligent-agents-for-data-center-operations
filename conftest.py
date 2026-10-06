"""Repository-wide pytest path setup for simulator and orchestrator tests."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
DC_TWIN_APP_DIR = REPO_ROOT / "aiopslab-applications" / "dataCenterTwin" / "app"

for path in (REPO_ROOT, DC_TWIN_APP_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
APP_ROOT = Path(__file__).resolve().parents[1]

for path in (REPO_ROOT, APP_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

# Makes `src/` importable from tests (so `import spc` works).
# pytest auto-discovers conftest.py and runs it before collecting tests.
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

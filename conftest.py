"""Makes `import contextfuse` work in tests without a packaging step."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

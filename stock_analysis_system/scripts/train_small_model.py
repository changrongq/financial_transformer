from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hierarchical_system.training import train_layer_model


if __name__ == "__main__":
    print(train_layer_model(str(ROOT / "config.yaml"), "small"))

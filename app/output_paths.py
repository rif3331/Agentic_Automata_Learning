from pathlib import Path

_OUTPUT_DIR = Path("runs")


def set_output_dir(path):
    global _OUTPUT_DIR
    _OUTPUT_DIR = Path(path).resolve()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_output_dir():
    return _OUTPUT_DIR


def get_artifact_dir(name):
    path = _OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
from pathlib import Path

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
RUN_DIR = ARTEFACTS_DIR


def initialize_run_directory(run_name: str) -> None:
    global RUN_DIR

    RUN_DIR = ARTEFACTS_DIR / run_name

    (RUN_DIR / "models").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "plots").mkdir(parents=True, exist_ok=True)


def get_run_directory() -> Path:
    return RUN_DIR


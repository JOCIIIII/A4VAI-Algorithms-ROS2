import os
from pathlib import Path

from .. import settings


def default_model_path() -> Path:
    directory = Path(os.environ.get('T1000_MODEL_DIR', Path(__file__).resolve().parents[2] / 'models'))
    return directory / settings.DEFAULT_POLICY

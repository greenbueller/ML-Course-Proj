"""
Kaggle authentication module

Loads your credentials from .env
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

def get_kaggle_api() -> KaggleApi:
    "Load Kaggle credentials and return authenticated Kaggle API instance."

    load_dotenv(ENV_FILE)

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise RuntimeError(
            "Kaggle credentials not found. \n\n"
            f"Expected them in: {ENV_FILE}\n\n"
            "Your .env file should contain: \n"
            "KAGGLE_USERNAME=your_username\n"
            "KAGGLE_KEY=your_key"
        )

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    api = KaggleApi()
    api.authenticate()

    return api
"""Kaggle authentication using the user's standard local credentials."""

from kaggle.api.kaggle_api_extended import KaggleApi

def get_kaggle_api() -> KaggleApi:
    """Load credentials from ~/.kaggle/kaggle.json and authenticate."""
    api = KaggleApi()
    api.authenticate()

    return api
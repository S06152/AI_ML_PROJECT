import os
from dotenv import load_dotenv

class EnvironmentConfig:
    """Handles environment variable loading."""

    def __init__(self):
        load_dotenv()

    @staticmethod
    def get(key: str) -> str:
        return os.getenv(key)

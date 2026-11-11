import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """One TestClient for the whole test session -- app startup (which loads
    both the Burgers and Darcy checkpoints) only needs to run once."""
    with TestClient(app) as c:
        yield c

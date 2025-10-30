# tests/conftest.py
import os
import pytest
from fastapi.testclient import TestClient

# Asegura que el código del backend esté en el path
os.environ.setdefault("PYTHONPATH", "backend/src")  # opcional, Makefile ya lo exporta

from neurocampus.app.main import app  # noqa: E402

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

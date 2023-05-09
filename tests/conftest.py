import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_path() -> Path:
    current_path = Path(__file__).parent.resolve()
    return current_path
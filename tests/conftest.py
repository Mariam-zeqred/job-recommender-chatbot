import pytest

from jobrec.data import load_postings


@pytest.fixture(scope="session")
def postings():
    return load_postings()

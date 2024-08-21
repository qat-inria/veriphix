import pytest
from numpy.random import PCG64, Generator

SEED = 25


@pytest.fixture()
def fx_rng() -> Generator:
    return Generator(PCG64(SEED))

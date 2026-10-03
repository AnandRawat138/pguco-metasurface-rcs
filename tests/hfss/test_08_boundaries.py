"""HFSS validation suite, step 08: Registered boundaries exist."""
from tests.hfss.conftest import check_row


def test_08_boundaries(verification):
    check_row(verification, "08")

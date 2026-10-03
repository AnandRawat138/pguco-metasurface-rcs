"""HFSS validation suite, step 11: Requested outputs are extracted, finite, on the requested grid and passive."""
from tests.hfss.conftest import check_row


def test_11_extraction(verification):
    check_row(verification, "11")

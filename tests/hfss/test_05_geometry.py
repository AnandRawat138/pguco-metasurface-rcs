"""HFSS validation suite, step 05: AEDT design validation passes after the variable update."""
from tests.hfss.conftest import check_row


def test_05_geometry(verification):
    check_row(verification, "05")

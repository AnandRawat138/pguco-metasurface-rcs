"""HFSS validation suite, step 01: PyAEDT imports and AEDT starts with the template project."""
from tests.hfss.conftest import check_row


def test_01_connection(verification):
    check_row(verification, "01")

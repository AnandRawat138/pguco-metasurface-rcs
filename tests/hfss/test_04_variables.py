"""HFSS validation suite, step 04: Every mapped design variable exists, is set with units and reads back."""
from tests.hfss.conftest import check_row


def test_04_variables(verification):
    check_row(verification, "04")

"""HFSS validation suite, step 06: Registered materials exist in the project."""
from tests.hfss.conftest import check_row


def test_06_materials(verification):
    check_row(verification, "06")

"""HFSS validation suite, step 10: Every requested incidence condition solves."""
from tests.hfss.conftest import check_row


def test_10_solve(verification):
    check_row(verification, "10")

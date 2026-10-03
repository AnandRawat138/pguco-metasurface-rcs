"""HFSS validation suite, step 12: Result package written and AEDT released."""
from tests.hfss.conftest import check_row


def test_12_cleanup(verification):
    check_row(verification, "12")

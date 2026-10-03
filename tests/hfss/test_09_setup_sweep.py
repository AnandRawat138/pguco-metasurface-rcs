"""HFSS validation suite, step 09: Setup and sweep exist; requested frequency sweep is applied and reads back."""
from tests.hfss.conftest import check_row


def test_09_setup_sweep(verification):
    check_row(verification, "09")

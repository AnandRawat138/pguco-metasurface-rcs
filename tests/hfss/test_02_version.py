"""HFSS validation suite, step 02: AEDT version matches the version the job was built for."""
from tests.hfss.conftest import check_row


def test_02_version(verification):
    check_row(verification, "02")

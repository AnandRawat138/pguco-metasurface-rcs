"""HFSS validation suite, step 03: Template file matches its registered SHA-256; design and solution type match."""
from tests.hfss.conftest import check_row


def test_03_open_template(verification):
    check_row(verification, "03")

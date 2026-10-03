"""HFSS validation suite, step 07: Registered excitations and angle/polarization controls exist and match the requested excitation."""
from tests.hfss.conftest import check_row


def test_07_excitations(verification):
    check_row(verification, "07")

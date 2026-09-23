import pytest
from fastapi.testclient import TestClient

def test_syntax_and_import():
    # Verify provisioning scripts compile cleanly
    import py_compile
    py_compile.compile("scripts/provision_verda.py", doraise=True)
    assert True

import pytest
from fastapi.testclient import TestClient

def test_syntax_and_import():
    # Verify scripts/provision_runpod.py compiles cleanly
    import py_compile
    py_compile.compile("scripts/provision_runpod.py", doraise=True)
    assert True

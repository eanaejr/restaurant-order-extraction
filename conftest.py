# Makes bare `pytest` work from the project root, same as `python -m pytest`:
# pytest imports root-level conftest.py first, which puts the project root on
# sys.path — so `from app import ...` inside tests/test_order.py resolves even
# without the `-m` flag. Nothing else lives here on purpose.

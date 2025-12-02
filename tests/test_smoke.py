"""Smoke tests to verify basic imports and setup work."""
import pytest

def test_can_import_litesearch():
    """Verify litesearch can be imported."""
    import litesearch
    assert hasattr(litesearch, '__version__')

def test_setup_db_creates_database(mem_db):
    """Verify setup_db returns a database object."""
    assert mem_db is not None

def test_mk_store_creates_table(mem_db):
    """Verify mk_store creates a content table."""
    store = mem_db.mk_store('test_content')
    # Verify table exists by querying sqlite_master
    tables = list(mem_db.query("SELECT name FROM sqlite_master WHERE type='table'"))
    table_names = [t['name'] for t in tables]
    assert 'test_content' in table_names

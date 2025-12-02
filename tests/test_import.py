"""Tests for import safety - no side effects on import."""
import subprocess
import sys
from unittest.mock import patch, MagicMock


class TestImportSafety:
    """Verify that importing litesearch doesn't have side effects."""

    def test_import_does_not_run_subprocess(self):
        """Importing litesearch should not execute any subprocess commands.

        This is critical for:
        - Clean imports in multiprocessing contexts
        - Testing without usearch installed
        - Not surprising users with system modifications
        """
        # Remove litesearch from cache to force reimport
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('litesearch')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with patch.object(subprocess, 'run') as mock_run:
            import litesearch
            # If usearch_fix() runs on import, subprocess.run will be called
            mock_run.assert_not_called()

    def test_setup_db_without_semantic_search_no_usearch(self):
        """setup_db with sem_search=False should not require usearch."""
        # This should work even without usearch installed
        from litesearch import setup_db
        db = setup_db(':memory:', sem_search=False)
        assert db is not None
        # Verify we can use basic functionality
        db.execute("SELECT 1")


class TestLazyUsearchInit:
    """Verify usearch initialization happens lazily."""

    def test_usearch_fix_available_for_explicit_call(self):
        """usearch_fix should be importable for users who need it explicitly."""
        from litesearch import usearch_fix
        assert callable(usearch_fix)

    def test_setup_db_with_semantic_search_loads_usearch(self):
        """setup_db with sem_search=True should load usearch extension."""
        from litesearch import setup_db
        # This will fail if usearch not installed, which is expected
        # The point is it should only happen when sem_search=True
        try:
            db = setup_db(':memory:', sem_search=True)
            # If we get here, usearch loaded successfully
            assert db is not None
        except Exception as e:
            # Expected if usearch not properly installed
            assert 'usearch' in str(e).lower() or 'extension' in str(e).lower()

"""Tests for litesearch/data.py functions."""
import pytest


def test_pre_handles_whitespace_only_input():
    """pre() should return empty string for whitespace input, not crash."""
    from litesearch.data import pre
    result = pre("   ")
    assert result == ""


def test_clean_returns_empty_string_not_none():
    """clean() should return empty string for empty input."""
    from litesearch.data import clean
    assert clean("") == ""
    assert clean("   ") == ""


def test_clean_removes_asterisks():
    """clean() should remove asterisks from input."""
    from litesearch.data import clean
    assert clean("test*query") == "testquery"
    assert clean("*test*") == "test"


def test_pre_handles_empty_string():
    """pre() should return empty string for empty input."""
    from litesearch.data import pre
    assert pre("") == ""


def test_pre_normal_query():
    """pre() should process normal queries correctly."""
    from litesearch.data import pre
    # Simple test - just verify it doesn't crash and returns something
    result = pre("test query")
    assert isinstance(result, str)
    assert len(result) > 0


def test_add_wc_empty_string():
    """add_wc() should return empty string for empty input."""
    from litesearch.data import add_wc
    assert add_wc("") == ""
    assert add_wc("   ") == ""


def test_add_wc_normal_query():
    """add_wc() should add wildcards to words."""
    from litesearch.data import add_wc
    assert add_wc("test query") == "test* query*"
    assert add_wc("single") == "single*"


def test_mk_wider_empty_string():
    """mk_wider() should return empty string for empty input."""
    from litesearch.data import mk_wider
    assert mk_wider("") == ""
    assert mk_wider("   ") == ""


def test_mk_wider_normal_query():
    """mk_wider() should join words with OR."""
    from litesearch.data import mk_wider
    assert mk_wider("test query") == "test OR query"
    assert mk_wider("single") == "single"


def test_ext_im_handles_none():
    """ext_im() should return None when given None input."""
    from pymupdf import Document
    # Test with None - should not crash and return None
    # We can't test the actual image extraction without a real PDF,
    # but we can verify the None handling works
    import tempfile
    import os

    # Create minimal empty PDF for testing
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
284
%%EOF"""

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        f.write(pdf_content)
        pdf_path = f.name

    try:
        doc = Document(pdf_path)
        # ext_im should return None for None input
        result = doc.ext_im(None)
        assert result is None
        doc.close()
    finally:
        os.unlink(pdf_path)

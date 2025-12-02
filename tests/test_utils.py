"""Test utils.py error handling - TDD approach for FastEncode silent failures."""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import numpy as np


class TestFastEncodeErrorHandling:
    """Test FastEncode raises errors instead of silent failures."""

    def test_raises_on_download_failure(self):
        """FastEncode should raise RuntimeError on download failure."""
        with patch('litesearch.utils.download_model') as mock_dl:
            mock_dl.side_effect = Exception("Network error")

            with pytest.raises(RuntimeError, match="Failed to download model"):
                from litesearch.utils import FastEncode
                FastEncode(repo_id="fake/model", model_dict=None)

    def test_raises_on_session_init_failure(self):
        """FastEncode should raise RuntimeError if ONNX session fails to initialize."""
        with patch('litesearch.utils.download_model') as mock_dl:
            mock_dl.return_value = "/tmp/fake_model"

            # Patch Path to make file exist check pass, then fail on session creation
            with patch('litesearch.utils.ort.InferenceSession') as mock_sess:
                mock_sess.side_effect = Exception("Invalid ONNX model")

                with pytest.raises(RuntimeError, match="Failed to initialize ONNX session"):
                    from litesearch.utils import FastEncode
                    FastEncode(repo_id="fake/model", model_dict=None)

    def test_encode_raises_if_session_none(self):
        """encode() should raise RuntimeError if ONNX session not initialized."""
        from litesearch.utils import FastEncode

        # Create instance with mocked download
        with patch('litesearch.utils.download_model') as mock_dl:
            mock_dl.return_value = "/tmp/fake_model"

            # Force session init to fail by making ort.InferenceSession raise
            with patch('litesearch.utils.ort.InferenceSession') as mock_sess:
                mock_sess.side_effect = Exception("Session init failed")

                # This should now raise during __init__ after the fix
                # But for testing encode() behavior, we need to manually set sess to None
                try:
                    encoder = FastEncode(repo_id="fake/model", model_dict=None)
                except RuntimeError:
                    # Expected after fix - create a minimal encoder with sess=None
                    encoder = FastEncode.__new__(FastEncode)
                    encoder.sess = None
                    encoder.dtype = np.float16

                    with pytest.raises(RuntimeError, match="ONNX session not initialized"):
                        encoder.encode(["test text"])

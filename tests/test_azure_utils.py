# tests/test_azure_utils.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import os
from azure.core.exceptions import ResourceNotFoundError
import importlib

# Import the module we're testing. Pytest path handling will find it.
import azure_utils

# Test the case where the environment variable is NOT set
@patch.dict(os.environ, {}, clear=True)
def test_get_blob_service_client_not_set():
    """ Tests that the client is None when the connection string is missing. """
    # Reload the module to ensure it runs its init code with the empty environment
    importlib.reload(azure_utils)
    assert azure_utils.get_blob_service_client() is None
    assert azure_utils.BLOB_SERVICE_CLIENT is None

# Test the case where the environment variable IS set
@patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "dummy_connection_string"})
@patch("azure.storage.blob.BlobServiceClient")
def test_azure_functions_all(MockBlobServiceClient):
    """
    This single test function covers all functions in azure_utils
    by setting up a comprehensive mock environment.
    """
    # Reload the module to make sure it picks up the patched env var
    # and initializes BLOB_SERVICE_CLIENT with the mock.
    importlib.reload(azure_utils)

    # --- Setup Mocks for the reloaded module ---
    # The patch already replaced BlobServiceClient, and from_connection_string is called
    # during the reload. We can now control its return value.
    mock_blob_service_client = MockBlobServiceClient.from_connection_string.return_value
    mock_blob_client = MagicMock()
    mock_blob_service_client.get_blob_client.return_value = mock_blob_client

    # --- Test download_df_from_blob (Success) ---
    df_content = "col1,col2\n1,a\n2,b"
    mock_downloader = MagicMock()
    mock_downloader.readall.return_value = df_content.encode('utf-8')
    mock_blob_client.download_blob.return_value = mock_downloader

    df = azure_utils.download_df_from_blob("container", "blob.csv")
    assert isinstance(df, pd.DataFrame)
    assert df['col1'].iloc[0] == 1
    mock_blob_service_client.get_blob_client.assert_called_with(container="container", blob="blob.csv")

    # --- Test download_df_from_blob (ResourceNotFoundError) ---
    mock_blob_client.download_blob.side_effect = ResourceNotFoundError
    empty_df = azure_utils.download_df_from_blob("container", "not_found.csv")
    assert empty_df.empty
    mock_blob_client.download_blob.side_effect = None # Reset side effect

    # --- Test upload_df_to_blob ---
    sample_df = pd.DataFrame({"test": [1]})
    azure_utils.upload_df_to_blob(sample_df, "container", "upload.csv")
    mock_blob_client.upload_blob.assert_called_once_with(sample_df.to_csv(index=False, encoding='utf-8'), overwrite=True)
    mock_blob_client.upload_blob.reset_mock()

    # --- Test download_file_from_blob ---
    with patch("builtins.open", mock_open()) as mock_file:
        azure_utils.download_file_from_blob("container", "file.txt", "local.txt")
        mock_file.assert_called_once_with("local.txt", "wb")
        mock_file().write.assert_called_once()

    # --- Test upload_file_to_blob ---
    with patch("builtins.open", mock_open(read_data=b"data")) as mock_file:
        azure_utils.upload_file_to_blob("local.txt", "container", "remote.txt")
        mock_file.assert_called_once_with("local.txt", "rb")
        mock_blob_client.upload_blob.assert_called_once()
        mock_blob_client.upload_blob.reset_mock()

    # --- Test upload_joblib_to_blob ---
    with patch("azure_utils.upload_file_to_blob") as mock_upload:
        azure_utils.upload_joblib_to_blob({"a": 1}, "container", "model.joblib")
        mock_upload.assert_called_once()
        temp_file_path = mock_upload.call_args.args[0]
        assert not os.path.exists(temp_file_path)

    # --- Test upload_json_to_blob ---
    with patch("azure_utils.upload_file_to_blob") as mock_upload:
        azure_utils.upload_json_to_blob({"key": "val"}, "container", "data.json")
        mock_upload.assert_called_once()
        temp_file_path = mock_upload.call_args.args[0]
        assert not os.path.exists(temp_file_path)
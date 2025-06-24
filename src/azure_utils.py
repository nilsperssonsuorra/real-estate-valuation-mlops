import os
import pandas as pd
import joblib
import json
import tempfile
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# This function will initialize the client only if the connection string is available
def get_blob_service_client():
    """Initializes and returns a BlobServiceClient if env var is set, else None."""
    connect_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if connect_str:
        return BlobServiceClient.from_connection_string(connect_str)
    return None

BLOB_SERVICE_CLIENT = get_blob_service_client()

# --- DataFrame Functions ---
def download_df_from_blob(container_name: str, blob_name: str) -> pd.DataFrame:
    """Downloads a blob and reads it into a pandas DataFrame."""
    from io import StringIO
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=blob_name)
    try:
        downloader = blob_client.download_blob()
        blob_contents = downloader.readall()
        return pd.read_csv(StringIO(blob_contents.decode('utf-8')))
    except ResourceNotFoundError:
        print(f"WARN: Blob '{blob_name}' not found in container '{container_name}'. Returning empty DataFrame.")
        return pd.DataFrame()

def upload_df_to_blob(df: pd.DataFrame, container_name: str, blob_name: str):
    """Uploads a pandas DataFrame to a blob."""
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=blob_name)
    output = df.to_csv(index=False, encoding='utf-8')
    blob_client.upload_blob(output, overwrite=True)
    print(f"Successfully uploaded DataFrame to {container_name}/{blob_name}")

# --- Generic File/Object Functions ---

def upload_file_to_blob(local_file_path: str, container_name: str, blob_name: str):
    """Uploads a local file to a blob."""
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=blob_name)
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Successfully uploaded {local_file_path} to {container_name}/{blob_name}")

def download_file_from_blob(container_name: str, blob_name: str, local_file_path: str):
    """Downloads a blob to a local file path."""
    blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=blob_name)
    with open(local_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    print(f"Successfully downloaded {blob_name} to {local_file_path}")

# --- Specialized Joblib/JSON Functions for convenience ---

def upload_joblib_to_blob(obj: any, container_name: str, blob_name: str):
    """Saves a Python object as a joblib file and uploads to a blob."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as temp_file:
        joblib.dump(obj, temp_file.name)
        upload_file_to_blob(temp_file.name, container_name, blob_name)
    os.remove(temp_file.name) # Clean up the temp file

def upload_json_to_blob(data: dict or list, container_name: str, blob_name: str):
    """Saves a dictionary or list as a JSON file and uploads to a blob."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", encoding='utf-8') as temp_file:
        json.dump(data, temp_file, ensure_ascii=False, indent=4)
        temp_file_path = temp_file.name
    # Must close the file before uploading
    upload_file_to_blob(temp_file_path, container_name, blob_name)
    os.remove(temp_file_path) # Clean up the temp file
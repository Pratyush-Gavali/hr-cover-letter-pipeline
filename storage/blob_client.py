# storage/blob_client.py  — full replacement

from __future__ import annotations
import hashlib
import os
from datetime import datetime, timedelta, timezone

from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
    ContentSettings,
)

MIME_TO_EXT = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}

# Azurite's hardcoded dev account key (public, not a secret)
_AZURITE_ACCOUNT_KEY = (
    "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
)


class CoverLetterBlobClient:
    """
    Azure Blob Storage client.
    Uses connection string auth — works with both Azurite (local)
    and real Azure Storage (set AZURE_STORAGE_CONNECTION_STRING in env).
    """

    def __init__(self, container_name: str = "covers"):
        conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        self._service = BlobServiceClient.from_connection_string(conn_str)
        self._container = container_name
        self._ensure_container()

    def _ensure_container(self) -> None:
        """Create the covers container if it doesn't exist yet."""
        try:
            self._service.create_container(self._container)
        except Exception:
            pass  # Already exists — fine

    def upload_raw(
        self,
        file_bytes: bytes,
        job_id: str,
        applicant_id: str,
        content_type: str,
    ) -> tuple[str, str]:
        ext = MIME_TO_EXT.get(content_type, "bin")
        blob_path = f"{job_id}/{applicant_id}/raw.{ext}"
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        client = self._service.get_blob_client(
            container=self._container, blob=blob_path
        )
        client.upload_blob(
            file_bytes,
            overwrite=True,          # Overwrite on re-upload (dev convenience)
            content_settings=ContentSettings(content_type=content_type),
            metadata={
                "applicant_id": applicant_id,
                "job_id": job_id,
                "sha256": content_hash,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return blob_path, content_hash

    def upload_masked(self, masked_text: str, job_id: str, applicant_id: str) -> str:
        blob_path = f"{job_id}/{applicant_id}/masked.txt"
        client = self._service.get_blob_client(
            container=self._container, blob=blob_path
        )
        client.upload_blob(
            masked_text.encode("utf-8"),
            overwrite=True,
            content_settings=ContentSettings(content_type="text/plain"),
        )
        return blob_path

    def download_raw(self, job_id: str, applicant_id: str, ext: str = "pdf") -> bytes:
        blob_path = f"{job_id}/{applicant_id}/raw.{ext}"
        return self._service.get_blob_client(
            container=self._container, blob=blob_path
        ).download_blob().readall()

    def generate_sas_url(self, blob_path: str, expiry_hours: int = 1) -> str:
        """
        SAS URL generation works identically against Azurite.
        Use the hardcoded Azurite account key for signing.
        """
        account_name = "devstoreaccount1"
        account_key = os.environ.get("AZURITE_ACCOUNT_KEY", _AZURITE_ACCOUNT_KEY)

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=self._container,
            blob_name=blob_path,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
        )
        base = os.environ.get(
            "AZURE_BLOB_ACCOUNT_URL",
            f"http://127.0.0.1:10000/{account_name}"
        )
        return f"{base}/{self._container}/{blob_path}?{sas_token}"

    def exists(self, job_id: str, applicant_id: str, ext: str = "pdf") -> bool:
        blob_path = f"{job_id}/{applicant_id}/raw.{ext}"
        return self._service.get_blob_client(
            container=self._container, blob=blob_path
        ).exists()
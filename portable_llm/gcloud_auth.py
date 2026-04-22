"""Google Cloud authentication helpers for Gemini Vertex AI."""

import logging
import os
import shutil
import subprocess
import time
from typing import ClassVar

import requests

logger = logging.getLogger(__name__)


class GCloudAuth:
    """Manages gcloud OAuth tokens and project resolution for Gemini Vertex AI."""

    _gemini_session: ClassVar[requests.Session | None] = None
    _gemini_access_token: ClassVar[str | None] = None
    _gemini_access_token_expiry: ClassVar[float] = 0.0
    _gemini_access_token_source: ClassVar[str | None] = None
    _gemini_project_id: ClassVar[str | None] = None
    _gcloud_executable: ClassVar[str | None] = None

    @classmethod
    def get_gemini_session(cls) -> requests.Session:
        if cls._gemini_session is None:
            cls._gemini_session = requests.Session()
            cls._gemini_session.headers.update(
                {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )
        return cls._gemini_session

    @classmethod
    def resolve_gemini_project_id(cls, config: dict) -> str:
        if cls._gemini_project_id:
            return cls._gemini_project_id

        project_id = (
            config.get("vertex", {}).get("project_id")
            or config.get("gemini", {}).get("project_id")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        if not project_id:
            result = subprocess.run(
                [cls.get_gcloud_executable(), "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=True,
            )
            project_id = result.stdout.strip()

        if not project_id or project_id == "(unset)":
            raise RuntimeError(
                "Gemini Vertex endpoint requires a GCP project ID. Set vertex.project_id or GOOGLE_CLOUD_PROJECT."
            )

        cls._gemini_project_id = project_id
        return project_id

    @classmethod
    def resolve_gemini_service_account(cls, config: dict) -> str | None:
        return (
            config.get("vertex", {}).get("service_account_email")
            or config.get("gemini", {}).get("service_account_email")
            or os.environ.get("GOOGLE_IMPERSONATE_SERVICE_ACCOUNT")
            or os.environ.get("VERTEX_SERVICE_ACCOUNT_EMAIL")
        )

    @classmethod
    def get_gemini_access_token(cls, config: dict) -> str:
        now = time.monotonic()
        service_account = cls.resolve_gemini_service_account(config)
        token_source = f"impersonate:{service_account}" if service_account else "gcloud-user"

        if (
            cls._gemini_access_token
            and now < cls._gemini_access_token_expiry
            and cls._gemini_access_token_source == token_source
        ):
            return cls._gemini_access_token

        command = [cls.get_gcloud_executable(), "auth", "print-access-token"]
        if service_account:
            command.append(f"--impersonate-service-account={service_account}")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            if not service_account:
                raise RuntimeError("Failed to obtain gcloud access token for Gemini Vertex endpoint.") from exc

            logger.warning(
                "Failed to impersonate service account '%s'; falling back to the active gcloud account. stderr=%s",
                service_account,
                (exc.stderr or "").strip(),
            )
            result = subprocess.run(
                [cls.get_gcloud_executable(), "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True,
            )
            token_source = "gcloud-user"

        token = result.stdout.strip()
        if not token:
            raise RuntimeError("Failed to obtain gcloud access token for Gemini Vertex endpoint.")

        cls._gemini_access_token = token
        cls._gemini_access_token_expiry = now + 300.0
        cls._gemini_access_token_source = token_source
        return token

    @classmethod
    def get_gcloud_executable(cls) -> str:
        if cls._gcloud_executable:
            return cls._gcloud_executable

        candidates = [
            shutil.which("gcloud.cmd"),
            shutil.which("gcloud"),
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud"),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                cls._gcloud_executable = candidate
                return candidate

        raise RuntimeError("gcloud CLI not found. Gemini Vertex endpoint requires gcloud authentication.")
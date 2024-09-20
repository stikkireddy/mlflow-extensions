import datetime
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import structlog
from databricks.sdk import WorkspaceClient
from structlog.stdlib import BoundLogger
from tenacity import retry, stop_after_attempt, wait_fixed

logger: BoundLogger = structlog.get_logger(__name__)


def full_volume_name_to_path(full_volume_name: str) -> Optional[str]:
    if full_volume_name is None:
        return None

    catalog_name: str
    schema_name: str
    volume_name: str
    parts: List[str] = full_volume_name.split(".")
    if len(parts) != 3:
        return None

    catalog_name, schema_name, volume_name = parts
    return f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"


class RotatingFileNamer:

    def __call__(self, default_name: str) -> str:
        name_parts = default_name.split(".")
        out_parts = name_parts[:-1]
        out_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        return ".".join(out_parts)


class FileRotator:

    def __call__(self, src: str, dst: str) -> None:
        if os.path.exists(src):
            os.rename(src, dst)


class VolumeRotator:

    def __init__(
        self,
        volume_path: str,
        databricks_host: Optional[str],
        databricks_token: Optional[str],
    ) -> None:

        self._volume_path = volume_path
        self._workspace_client: WorkspaceClient = WorkspaceClient(
            host=databricks_host, token=databricks_token
        )
        logger.info(f"Creating directory: {self._volume_path}")
        self._workspace_client.files.create_directory(self._volume_path)

    def __call__(self, src: str, dst: str) -> None:
        if os.path.exists(src):
            os.rename(src, dst)
            with open(dst, "r") as fin:
                content: str = fin.read()
                filename: str = Path(dst).name
                self._upload(
                    f"{self._volume_path}/{filename}.jsonl", content, overwrite=True
                )

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
    def _upload(
        self, file_path: str, contents: bytearray, overwrite: bool = True
    ) -> None:
        try:
            logger.info("Uploading log file", file_path=file_path)
            self._workspace_client.files.upload(file_path, contents, overwrite=True)
        except Exception as e:
            logger.error("Failed to upload file", file_path=file_path, error=str(e))


def _get_databricks_host_creds(
    databricks_host: Optional[str], databricks_token: Optional[str]
) -> Tuple[str, str]:
    if databricks_host is None or databricks_token is None:
        try:
            from mlflow.utils.databricks_utils import get_databricks_host_creds

            databricks_host = os.environ.get(
                "DATABRICKS_HOST", get_databricks_host_creds().host
            )
            databricks_token = os.environ.get(
                "DATABRICKS_TOKEN", get_databricks_host_creds().token
            )
        except Exception as e:
            databricks_host = os.environ.get("DATABRICKS_HOST", None)
            databricks_token = os.environ.get("DATABRICKS_TOKEN", None)
            logger.info("Failed to get databricks creds", error=str(e))

    return databricks_host, databricks_token


def create_rotator(
    volume_path: Optional[str],
    databricks_host: Optional[str],
    databricks_token: Optional[str],
) -> Callable[[str, str], None]:

    databricks_host, databricks_token = _get_databricks_host_creds(
        databricks_host, databricks_token
    )

    rotator: Callable[[str, str], None] = FileRotator()
    if all([v is not None for v in [databricks_host, databricks_token, volume_path]]):
        try:
            rotator = VolumeRotator(
                volume_path=volume_path,
                databricks_host=databricks_host,
                databricks_token=databricks_token,
            )
        except Exception as e:
            logger.error("Failed to create VolumeRotator", error=str(e))

    logger.info("Rotator created", rotator=type(rotator))
    return rotator


class SizeAndTimedRotatingVolumeHandler(TimedRotatingFileHandler):

    def __init__(
        self,
        filename: str,
        archive_path: Optional[str] = None,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
        when: str = "h",
        interval: int = 1,
        max_bytes: int = 0,
        backup_count: int = 0,
        encoding: Optional[str] = None,
        delay: bool = False,
        utc: bool = False,
        at_time=None,
        errors=None,
    ):
        if max_bytes > 0:
            max_bytes = max(1024, max_bytes)
        self._max_bytes = max_bytes

        parent: Path = Path(filename).parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        TimedRotatingFileHandler.__init__(
            self,
            filename,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=at_time,
            errors=errors,
        )
        self.backup_count = backup_count

        self.namer: Callable[[str], str] = RotatingFileNamer()
        self.rotator: Callable[[str, str], None] = create_rotator(
            archive_path, databricks_host, databricks_token
        )

    def shouldRolloverOnSize(self) -> bool:
        if os.path.exists(self.baseFilename) and not os.path.isfile(self.baseFilename):
            return False

        if self.stream is None:
            return False

        if self._max_bytes > 0:
            self.stream.seek(0, 2)
            if self.stream.tell() >= self._max_bytes:
                return True
        return False

    def shouldRollover(self, record: bytearray) -> bool:
        return self.shouldRolloverOnSize() or super().shouldRollover(record)

    def getFilesToDelete(self) -> List[str]:
        dir_name, base_name = os.path.split(self.baseFilename)
        files_dict = {}
        for fileName in os.listdir(dir_name):
            _, ext = os.path.splitext(fileName)
            date_str = ext.replace(".", "")
            try:
                d = datetime.strptime(date_str, "%Y%m%d_%H%M%S_%f")
                files_dict[d] = fileName
            except:
                pass
        if len(files_dict) < self.backup_count:
            return []

        sorted_dict = dict(sorted(files_dict.items(), reverse=True))
        return [os.path.join(dir_name, v) for k, v in sorted_dict.items()][
            self.backup_count :
        ]


def rotating_volume_handler(
    filename: str,
    archive_path: Optional[str] = None,
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    when: str = "h",
    interval: int = 1,
    max_bytes: int = 0,
    backup_count: int = 0,
    encoding: Optional[str] = None,
    delay: bool = False,
    utc: bool = False,
    at_time=None,
    errors=None,
) -> logging.Handler:
    return SizeAndTimedRotatingVolumeHandler(
        filename=filename,
        archive_path=archive_path,
        databricks_host=databricks_host,
        databricks_token=databricks_token,
        when=when,
        interval=interval,
        max_bytes=max_bytes,
        backup_count=backup_count,
        encoding=encoding,
        delay=delay,
        utc=utc,
        at_time=at_time,
        errors=errors,
    )

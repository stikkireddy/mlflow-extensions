import datetime
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, List, Optional

import structlog
from databricks.sdk import WorkspaceClient
from structlog.stdlib import BoundLogger
from tenacity import retry, stop_after_attempt, wait_fixed

logger: BoundLogger = structlog.get_logger(__name__)

DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT: int = 5


def full_volume_name_to_path(full_volume_name: str) -> str:
    catalog_name: str
    schema_name: str
    volume_name: str
    catalog_name, schema_name, volume_name = full_volume_name.split(".")
    return f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"


class RotatingFileNamer:

    def __call__(self, default_name: str) -> str:
        name_parts = default_name.split(".")
        out_parts = name_parts[:-1]
        out_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        return ".".join(out_parts)


class VolumeRotator:

    def __init__(
        self,
        volume_path: str,
        databricks_host: Optional[str],
        databricks_token: Optional[str],
    ) -> None:
        from mlflow.utils.databricks_utils import get_databricks_host_creds

        self._volume_path = volume_path
        if databricks_host is None or databricks_token is None:
            databricks_host = get_databricks_host_creds().host
            databricks_token = get_databricks_host_creds().token

        self._workspace_client: WorkspaceClient = WorkspaceClient(
            host=databricks_host, token=databricks_token
        )

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


class SizeAndTimedRotatingVolumeHandler(TimedRotatingFileHandler):

    def __init__(
        self,
        filename: str,
        volume_path: str,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
        when: str = "h",
        interval: int = 1,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        encoding: Optional[str] = None,
        delay: bool = False,
        utc: bool = False,
        at_time=None,
        errors=None,
    ):
        if max_bytes > 0:
            max_bytes = max(1024, max_bytes)
        self._max_bytes = max_bytes

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
        self.rotator: Callable[[str], str] = VolumeRotator(
            volume_path=volume_path,
            databricks_host=databricks_host,
            databricks_token=databricks_token,
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
                d = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
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
    volume_path: str,
    databricks_host: Optional[str] = None,
    databricks_token: Optional[str] = None,
    when: str = "h",
    interval: int = 1,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    encoding: Optional[str] = None,
    delay: bool = False,
    utc: bool = False,
    at_time=None,
    errors=None,
) -> SizeAndTimedRotatingVolumeHandler:
    return SizeAndTimedRotatingVolumeHandler(
        filename=filename,
        volume_path=volume_path,
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

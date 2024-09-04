import os
import subprocess
from dataclasses import dataclass


@dataclass
class DiagnosticCmd:
    key: str
    cmd: str
    desc: str


def run_cmd(cmd: DiagnosticCmd):
    try:
        # Run the command
        result = subprocess.run(
            cmd.cmd,
            shell=True,  # Use shell=True to allow command string input
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Ensure output is returned as a string
            preexec_fn=os.setsid,  # Run in a new session
        )

        # Return results
        return {
            "command": cmd.cmd,
            "command_desc": cmd.desc,
            "stdout": result.stdout.strip(),  # Strip to remove any extra newlines
            "stderr": result.stderr.strip(),  # Strip to remove any extra newlines
            "returncode": result.returncode,
        }

    except Exception as e:
        # Return an error message
        return {
            "command": cmd.cmd,
            "command_desc": cmd.desc,
            "stdout": None,  # Strip to remove any extra newlines
            "stderr": str(e),  # Strip to remove any extra newlines
            "returncode": -1,
        }


COMPUTE_DETAILS_COMMANDS = [
    DiagnosticCmd("nvidia-smi", "nvidia-smi", "NVIDIA System Management Interface"),
    DiagnosticCmd("nvidia-topology", "nvidia-smi topo -m", "NVIDIA GPU Topology"),
    DiagnosticCmd("disk-space", "df -h", "Disk Space"),
    DiagnosticCmd("current-user", "whoami", "Current User"),
    DiagnosticCmd("active-procs", "ps aux", "Processes"),
    DiagnosticCmd("check-shm", "touch /dev/shm/test_shm", "Shared Memory Test"),
    DiagnosticCmd("pip-list", "pip list", "Installed Python Packages"),
    DiagnosticCmd("python-version", "python --version", "Python Version"),
    DiagnosticCmd("os-details", "lsb_release -a", "Linux Distribution"),
    DiagnosticCmd("arch-details", "uname -m", "Machine Architecture"),
    DiagnosticCmd("env", "env", "Environment Variables"),
    DiagnosticCmd("cpu-count", "nproc", "Number of Processors"),
    DiagnosticCmd("memory-count", "free -h", "Memory Usage"),
]

COMPUTE_DETAILS_COMMANDS_MAP = {cmd.key: cmd for cmd in COMPUTE_DETAILS_COMMANDS}


def get_compute_details(cmd_key: str) -> dict:
    if cmd_key == "all":
        return {"compute_details": [run_cmd(cmd) for cmd in COMPUTE_DETAILS_COMMANDS]}
    elif cmd_key in COMPUTE_DETAILS_COMMANDS_MAP:
        return {"compute_details": run_cmd(COMPUTE_DETAILS_COMMANDS_MAP[cmd_key])}
    else:
        return {
            "compute_details": {
                "command": cmd_key,
                "stdout": None,
                "stderr": f"Command {cmd_key} not found",
                "returncode": -1,
            }
        }

import socket
import threading
import typing
from typing import List

from mlflow_extensions.databricks.deploy.ez_deploy import EzDeployConfig

if typing.TYPE_CHECKING is True:
    from mlflow.pyfunc import PythonModelContext


def make_process_and_get_artifacts(config: EzDeployConfig, local_dir=None):
    if local_dir is not None:
        artifacts = config.engine_config.setup_artifacts(local_dir)
    else:
        artifacts = config.engine_config.setup_artifacts()

    engine = config.engine_proc(config=config.engine_config)

    return engine, artifacts


def force_on_node(node_id: str, remote_func_or_actor_class):
    import ray

    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id, soft=False
    )
    options = {"scheduling_strategy": scheduling_strategy}
    return remote_func_or_actor_class.options(**options)


def run_on_every_node(remote_func_or_actor_class, **remote_kwargs):
    import ray

    refs = []
    for node in ray.nodes():
        if node["Alive"] and node["Resources"].get("GPU", None):
            refs.append(
                force_on_node(node["NodeID"], remote_func_or_actor_class).remote(
                    **remote_kwargs
                )
            )
    return ray.get(refs)


def parse_vllm_configs(
    config: EzDeployConfig, node_info: List, ctx: "PythonModelContext"
):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser

    vllm_comf = config.engine_config._to_vllm_command(ctx)[3:]
    for index, arg in enumerate(vllm_comf):
        if type(arg) != str:
            vllm_comf[index] = str(arg)

    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    parsed_args = parser.parse_args(args=vllm_comf)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.tensor_parallel_size = node_info[0].gpu_count
    tp = engine_args.tensor_parallel_size
    print(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 4})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors
    return pg_resources, parsed_args, engine_args


def block_port(port, host="0.0.0.0"):
    """
    Starts a shadow process that binds to a specified port to block it from being used by other processes.

    Args:
        port (int): The port to block.
        host (str): The host to bind to (default is '0.0.0.0' which binds to all interfaces).

    Returns:
        tuple: A tuple containing the thread, socket object, and a stop event (shadow_thread, shadow_socket, stop_event).
    """
    # Create a socket to bind to the port
    shadow_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    shadow_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    shadow_socket.bind((host, port))
    shadow_socket.listen(5)

    stop_event = threading.Event()  # Event to signal the server to stop

    def shadow_server(sock, stop_event):
        print(f"Shadow process started and port {port} is now blocked.")
        try:
            # Keep the socket open to block the port
            while not stop_event.is_set():
                sock.settimeout(
                    1.0
                )  # Set a timeout to allow checking for the stop event
                try:
                    conn, _ = sock.accept()  # Accept incoming connections
                    conn.close()
                except socket.timeout:
                    # Continue the loop if a timeout occurs (to periodically check for the stop event)
                    continue
        except Exception as e:
            print(f"Shadow process for port {port} stopped: {e}")
        finally:
            sock.close()

    # Start the shadow server in a separate thread
    shadow_thread = threading.Thread(
        target=shadow_server, args=(shadow_socket, stop_event), daemon=True
    )
    shadow_thread.start()

    return shadow_thread, shadow_socket, stop_event


def unblock_port(shadow_socket, stop_event):
    """
    Unblocks the port by closing the shadow socket and stopping the shadow server.

    Args:
        shadow_socket (socket.socket): The socket that is blocking the port.
        stop_event (threading.Event): The event to signal stopping the server.
    """
    # Signal the thread to stop
    stop_event.set()

    # Close the socket to unblock the port
    shadow_socket.close()
    print("Port unblocked and shadow server stopped.")

# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Coordinator Client."""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time
from typing import Any

from ..nccl.mock_bootstrap import prepare_mock_bootstrap_redis
from ..runner import AssistantRunner, RunnerManager, TrainRunner
from .protocol import (
    Message,
    MessageType,
    ReplyCode,
    create_client_hello,
    create_client_reply,
    decode_message,
    describe_reply_code,
    encode_message,
)

logger = logging.getLogger(__name__)


class CrystalClient:
    """Coordinator Client for connecting to and communicating with the server.

    The client connects to a Coordinator Server, registers itself with a rank,
    and listens for commands from the server.

    Attributes:
        host: The server host address
        port: The server port
        rank: The client rank identifier (also used as node_id)
        crystal_workspace: Absolute path to the crystal workspace directory
    """

    def __init__(self, host: str, port: int, rank: int, crystal_workspace: str) -> None:
        """Initialize the CrystalClient.

        Args:
            host: The server host address
            port: The server port
            rank: The client rank identifier (also used as node_id)
            crystal_workspace: Absolute path to the crystal workspace directory
        """
        self.host = host
        self.port = port
        self.rank = rank
        self.crystal_workspace = crystal_workspace

        logger.info(
            "Initializing CrystalClient with host=%s, port=%s, rank=%s, crystal_workspace=%s",
            host,
            port,
            rank,
            crystal_workspace,
        )

        self._running = False
        self._socket: socket.socket | None = None

    def start(self) -> None:
        """Start the client and begin listening for messages."""
        self._running = True
        logger.info("Client rank %s started", self.rank)

        try:
            while self._running:
                try:
                    self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._socket.connect((self.host, self.port))

                    self._send_hello()
                    logger.info("Connected to server at %s:%s", self.host, self.port)

                    self._process_messages()
                except OSError:
                    if self._running and self._socket:
                        _socket = self._socket
                        with contextlib.suppress(OSError):
                            _socket.close()
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("Received keyboard interrupt")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the client gracefully."""
        logger.info("Stopping client...")
        if self._running:
            self._running = False

            if self._socket:
                _socket = self._socket
                with contextlib.suppress(OSError):
                    _socket.shutdown(socket.SHUT_RDWR)
                    _socket.close()
                self._socket = None

            logger.info("Disconnected from server")

    # -------------------------------------------------------------------------
    # Private Methods - Message Sending
    # -------------------------------------------------------------------------

    def _send_message(self, message: Message) -> None:
        """Send a message to the server.

        Args:
            message: The message to send
        """
        if not self._socket:
            logger.error("Cannot send message: not connected")
            return

        try:
            self._socket.send(encode_message(message))
            logger.info("Sent message: %s", message.type)
        except Exception:
            logger.exception("Failed to send message")

    def _send_hello(self) -> None:
        """Send a CLIENT_HELLO message."""
        hello = create_client_hello(self.rank)
        self._send_message(hello)

    def _send_reply(
        self,
        code: int,
        handle: str = "",
        msg: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Send a CLIENT_REPLY message.

        Args:
            code: A ReplyCode member, or the subprocess exit code that
                TRAIN_WAIT reports verbatim.
            handle: The message handle to match request and reply
            msg: Error message when code is non-zero (default: empty string)
            payload: Optional structured response payload.
        """
        reply = create_client_reply(code, self.rank, handle, msg, payload)
        self._send_message(reply)

    # -------------------------------------------------------------------------
    # Private Methods - Message Processing
    # -------------------------------------------------------------------------

    def _process_messages(self) -> None:
        """Process messages from the server.

        This method runs in a loop receiving and handling messages from the
        server until the connection is closed or the client is stopped.
        """
        buffer = ""

        while self._running and self._socket:
            try:
                chunk = self._socket.recv(4096).decode("utf-8")
                if not chunk:
                    logger.info("Connection closed by server")
                    break

                buffer += chunk

                # Process complete messages
                while "\n" in buffer:
                    message_str, buffer = buffer.split("\n", 1)
                    if message_str.strip():
                        self._handle_message(message_str)
            except OSError as e:
                if self._running:
                    logger.error("Error receiving message: %s", e)
                    raise e
                break

    def _handle_message(self, data: str) -> None:
        """Handle a message from the server.

        Args:
            data: The raw message data
        """
        try:
            message = decode_message(data)
            logger.info("Received message: %s with data %s", message.type, message.data)

            if message.type == MessageType.SHUTDOWN_CLIENT:
                self._handle_shutdown(message)
            elif message.type == MessageType.CLIENT_PRINT:
                self._handle_print(message)
            elif message.type == MessageType.TRAIN_START:
                self._handle_train_start(message)
            elif message.type == MessageType.TRAIN_WAIT:
                self._handle_train_wait(message)
            elif message.type == MessageType.TRAIN_STOP:
                self._handle_train_stop(message)
            elif message.type == MessageType.MOCK_BOOTSTRAP_PREPARE:
                self._handle_mock_bootstrap_prepare(message)
            elif message.type == MessageType.ASSISTANT_START:
                self._handle_assistant_start(message)
            elif message.type == MessageType.ASSISTANT_STOP:
                self._handle_assistant_stop(message)
            else:
                logger.warning("Unknown message type: %s", message.type)
        except Exception:
            logger.exception("Error handling message")

    # -------------------------------------------------------------------------
    # Private Methods - Message Handlers
    # -------------------------------------------------------------------------

    def _handle_shutdown(self, message: Message) -> None:
        """Handle SHUTDOWN_CLIENT message.

        This method stops the client gracefully and disconnects from the server
        when a SHUTDOWN_CLIENT message is received.

        Args:
            message: The shutdown message
        """
        logger.info("Received shutdown command")
        self.stop()

    def _handle_print(self, message: Message) -> None:
        """Handle CLIENT_PRINT message.

        Args:
            message: The print message
        """
        message_str = message.data.get("str", "")
        handle = message.data.get("_handle", "")
        print(message_str)
        self._send_reply(ReplyCode.SUCCESS, handle)

    def _handle_train_start(self, message: Message) -> None:
        """Handle TRAIN_START message.

        Creates and starts a TrainRunner to execute the training script.

        Args:
            message: The TRAIN_START message containing train_script_path, extra_args_str, etc.
        """
        # Check if there is already an active runner
        if RunnerManager().get_active_runner() is not None:
            error_msg = "Runner quota already in use, cannot start training"
            logger.error(error_msg)
            self._send_reply(ReplyCode.RUNNER_QUOTA_IN_USE, message.data.get("_handle", ""), error_msg)
            return

        try:
            master_addr = message.data.get("master_addr", "")
            master_port = message.data.get("master_port", 0)
            node_rank = message.data.get("node_rank", "")
            log_dir_name = message.data.get("log_dir_name", "")
            train_script_path = message.data.get("train_script_path", "")
            extra_args_str = message.data.get("extra_args_str", "")
            extra_envs_str = message.data.get("extra_envs_str", "")

            # Create TrainRunner
            train_runner = TrainRunner(
                master_addr=master_addr,
                master_port=master_port,
                node_rank=node_rank,
                crystal_workspace=self.crystal_workspace,
                log_dir_name=log_dir_name,
                train_script_path=train_script_path,
                extra_args_str=extra_args_str,
                extra_envs_str=extra_envs_str,
            )

            # Start training
            train_runner.execute()

            # Send success reply
            self._send_reply(ReplyCode.SUCCESS, message.data.get("_handle", ""))
            logger.info("Training started successfully on node %s", self.rank)

        except Exception:
            logger.exception("Failed to start training")
            error_msg = "Failed to start training"
            self._send_reply(ReplyCode.INTERNAL_ERROR, message.data.get("_handle", ""), error_msg)

    def _handle_train_wait(self, message: Message) -> None:
        """Handle TRAIN_WAIT message.

        Waits for the training to complete and returns the result.
        This method uses a separate thread to avoid blocking the main message loop.

        Args:
            message: The TRAIN_WAIT message
        """
        train_runner = RunnerManager().get_active_runner()

        if train_runner is None or not isinstance(train_runner, TrainRunner):
            error_msg = "No active TrainRunner, cannot wait for training"
            logger.error(error_msg)
            self._send_reply(ReplyCode.NO_ACTIVE_RUNNER, message.data.get("_handle", ""), error_msg)
            return

        handle = message.data.get("_handle", "")

        def wait_callback(return_code: int | None) -> None:
            """Callback function called when training completes.

            Args:
                return_code: The return code from the training process, or
                    None when the runner could not wait for it.
            """
            if return_code is None:
                self._send_reply(ReplyCode.INTERNAL_ERROR, handle, "failed to wait for the training process")
            else:
                msg = "" if return_code == ReplyCode.SUCCESS else describe_reply_code(return_code)
                self._send_reply(return_code, handle, msg)
            logger.info("Training completed on node %s with return code %s", self.rank, return_code)

        try:
            # Start a separate thread to wait for training completion
            wait_thread = threading.Thread(
                target=train_runner.wait,
                args=(wait_callback,),
                daemon=True,
            )
            wait_thread.start()
            logger.info("Started wait thread for training on node %s", self.rank)

        except Exception:
            logger.exception("Error starting wait thread for training")
            error_msg = "Error starting wait thread for training"
            self._send_reply(ReplyCode.INTERNAL_ERROR, handle, error_msg)

    def _handle_train_stop(self, message: Message) -> None:
        """Handle TRAIN_STOP message.

        Forcefully stops the training process.

        Args:
            message: The TRAIN_STOP message
        """
        train_runner = RunnerManager().get_active_runner()

        if train_runner is None or not isinstance(train_runner, TrainRunner):
            logger.warning("No active TrainRunner to stop")
            self._send_reply(ReplyCode.SUCCESS, message.data.get("_handle", ""))
            return

        try:
            # Kill the training process
            train_runner.kill()

            # Send success reply
            self._send_reply(ReplyCode.SUCCESS, message.data.get("_handle", ""))
            logger.info("Training stopped successfully on node %s", self.rank)

        except Exception:
            logger.exception("Error stopping training")
            error_msg = "Error stopping training"
            self._send_reply(ReplyCode.INTERNAL_ERROR, message.data.get("_handle", ""), error_msg)

    def _handle_mock_bootstrap_prepare(self, message: Message) -> None:
        """Handle MOCK_BOOTSTRAP_PREPARE message."""
        if RunnerManager().get_active_runner() is not None:
            error_msg = "Runner quota already in use, cannot prepare mock bootstrap"
            logger.error(error_msg)
            self._send_reply(ReplyCode.RUNNER_QUOTA_IN_USE, message.data.get("_handle", ""), error_msg)
            return

        handle = message.data.get("_handle", "")

        def prepare_callback() -> None:
            """Prepare mock bootstrap Redis state and reply with payload."""
            try:
                payload = prepare_mock_bootstrap_redis(
                    redis_host=message.data["redis_host"],
                    trace_dir=message.data["trace_dir"],
                    sandbox_nodes=message.data["sandbox_nodes"],
                    nnodes=message.data["nnodes"],
                    gpus_per_node=message.data.get("gpus_per_node", 8),
                )
                self._send_reply(ReplyCode.SUCCESS, handle, payload=payload)
                logger.info("Mock bootstrap prepare completed successfully on node %s", self.rank)
            except Exception as e:
                logger.exception("Mock bootstrap prepare failed")
                self._send_reply(ReplyCode.INTERNAL_ERROR, handle, str(e))

        try:
            prepare_thread = threading.Thread(
                target=prepare_callback,
                daemon=True,
            )
            prepare_thread.start()
            logger.info("Started mock bootstrap prepare thread on node %s", self.rank)
        except Exception:
            logger.exception("Failed to start mock bootstrap prepare")
            error_msg = "Failed to start mock bootstrap prepare"
            self._send_reply(ReplyCode.INTERNAL_ERROR, handle, error_msg)

    def _handle_assistant_start(self, message: Message) -> None:
        """Handle ASSISTANT_START message.

        Creates and starts an AssistantRunner to launch assistant processes.

        Args:
            message: The ASSISTANT_START message containing sandbox_nodes, nnodes, etc.
        """
        # Check if there is already an active runner
        if RunnerManager().get_active_runner() is not None:
            error_msg = "Runner quota already in use, cannot start assistants"
            logger.error(error_msg)
            self._send_reply(ReplyCode.RUNNER_QUOTA_IN_USE, message.data.get("_handle", ""), error_msg)
            return

        try:
            master_addr = message.data.get("master_addr")
            master_port = message.data.get("master_port")
            redis_host = message.data["redis_host"]
            log_dir_name = message.data.get("log_dir_name")
            emulated_nodes_for_this_assistant = message.data["emulated_nodes_for_this_assistant"]
            nnodes = message.data.get("nnodes")
            sandbox_nodes = message.data.get("sandbox_nodes")
            assistant_envs_str = message.data.get("assistant_envs_str")
            trace_dir = message.data.get("trace_dir")
            graph_dir = message.data.get("graph_dir")
            graph_cache_path = message.data.get("graph_cache_path")
            profiler_dir = message.data.get("profiler_dir")
            pp_rank_stride = message.data.get("pp_rank_stride")
            enable_assistant_online_update = message.data.get("enable_assistant_online_update")
            tensor_provider = message.data.get("tensor_provider")
            enable_mock_nccl_bootstrap = message.data.get("enable_mock_nccl_bootstrap", False)
            assistant_nsight_system_ranks = message.data.get("assistant_nsight_system_ranks")
            gpus_per_node = message.data.get("gpus_per_node", 8)

            # Create AssistantRunner
            assistant_runner = AssistantRunner(
                node_id=self.rank,
                master_addr=master_addr,
                master_port=master_port,
                redis_host=redis_host,
                crystal_workspace=self.crystal_workspace,
                log_dir_name=log_dir_name,
                emulated_nodes_for_this_assistant=emulated_nodes_for_this_assistant,
                nnodes=nnodes,
                sandbox_nodes=sandbox_nodes,
                assistant_envs_str=assistant_envs_str,
                trace_dir=trace_dir,
                graph_dir=graph_dir,
                graph_cache_path=graph_cache_path,
                profiler_dir=profiler_dir,
                pp_rank_stride=pp_rank_stride,
                enable_assistant_online_update=enable_assistant_online_update,
                tensor_provider=tensor_provider,
                enable_mock_nccl_bootstrap=enable_mock_nccl_bootstrap,
                assistant_nsight_system_ranks=assistant_nsight_system_ranks,
                gpus_per_node=gpus_per_node,
            )

            # Launch assistants
            assistant_runner.execute()

            # Send success reply
            self._send_reply(ReplyCode.SUCCESS, message.data.get("_handle", ""))
            logger.info("Assistants started successfully on node %s", self.rank)

        except Exception:
            logger.exception("Failed to start assistants")
            error_msg = "Failed to start assistants"
            self._send_reply(ReplyCode.INTERNAL_ERROR, message.data.get("_handle", ""), error_msg)

    def _handle_assistant_stop(self, message: Message) -> None:
        """Handle ASSISTANT_STOP message.

        Forcefully stops the assistant processes.

        Args:
            message: The ASSISTANT_STOP message
        """
        assistant_runner = RunnerManager().get_active_runner()

        if assistant_runner is None or not isinstance(assistant_runner, AssistantRunner):
            logger.warning("No active AssistantRunner to stop")
            self._send_reply(ReplyCode.SUCCESS, message.data.get("_handle", ""))
            return

        try:
            # Kill the assistant processes
            assistant_runner.kill()

            # Send success reply
            self._send_reply(ReplyCode.SUCCESS, message.data.get("_handle", ""))
            logger.info("Assistants stopped successfully on node %s", self.rank)

        except Exception:
            logger.exception("Error stopping assistants")
            error_msg = "Error stopping assistants"
            self._send_reply(ReplyCode.INTERNAL_ERROR, message.data.get("_handle", ""), error_msg)

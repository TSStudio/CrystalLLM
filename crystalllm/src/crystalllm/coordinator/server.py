# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Coordinator Server."""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from .protocol import (
    Message,
    MessageType,
    ReplyCode,
    create_assistant_start,
    create_assistant_stop,
    create_client_print,
    create_mock_bootstrap_prepare,
    create_shutdown_client,
    create_train_start,
    create_train_stop,
    create_train_wait,
    decode_message,
    describe_reply_code,
    encode_message,
)

logger = logging.getLogger(__name__)

# Interval between connectivity checks while waiting for client replies, so a
# client that goes offline without replying is detected instead of blocking
# the waiter forever.
_REPLY_WAIT_POLL_INTERVAL_SECONDS = 1.0


class CrystalServer:
    """Coordinator Server for managing and communicating with clients.

    The server listens for incoming client connections, maintains a registry
    of connected clients, and provides methods for sending commands and
    collecting results.

    Attributes:
        host: The host address to bind to
        port: The port to listen on
    """

    def __init__(self, host: str, port: int) -> None:
        """Initialize the CrystalServer.

        Args:
            host: The host address to bind to
            port: The port to listen on
        """
        self.host = host
        self.port = port

        self._running = False
        self._server_socket: socket.socket | None = None
        self._start_time = 0.0

        # Client registry: {rank: {"socket": socket, "address": address}}
        self._clients: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._controlled_ranks: set[int] = set()

        # Read-write lock for thread-safe access to _clients
        self._clients_lock = threading.Lock()

        # Reply tracking: {handle: {rank: threading.Event}}
        self._reply_events: dict[str, dict[int, threading.Event]] = {}
        # Reply results: {handle: {rank: code}}, where code is int (0 for success, non-zero for failure)
        self._reply_results: dict[str, dict[int, int]] = {}
        # Reply payloads: {handle: {rank: payload}}, used by RPCs that return structured data.
        self._reply_payloads: dict[str, dict[int, Any]] = {}

        # Thread for accepting connections
        self._accept_thread: threading.Thread | None = None

    # -------------------------------------------------------------------------
    # Public Methods - Server Lifecycle
    # -------------------------------------------------------------------------

    def start_server(self) -> None:
        """Start the server and begin listening for connections.

        Raises:
            OSError: If the port is already in use or cannot be bound
        """
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1024)
        self._running = True
        self._start_time = time.time()

        logger.info("CrystalServer listening on %s:%s", self.host, self.port)

        # Start accept thread
        self._accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        self._accept_thread.start()

    def stop_server(self) -> None:
        """Stop the server and close all connections."""
        logger.info("CrystalServer is stopping...")
        self._running = False

        # Close all client connections
        for rank, client_info in list(self._clients.items()):
            sock = client_info.get("socket")
            if sock:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
            logger.info("Closed connection to client rank %s", rank)

        self._clients.clear()

        # Close server socket
        if self._server_socket:
            with contextlib.suppress(OSError):
                self._server_socket.shutdown(socket.SHUT_RDWR)
                self._server_socket.close()
            self._server_socket = None

        total_time = time.time() - self._start_time
        logger.info("Server ran for %.4fs", total_time)
        logger.info("CrystalServer stopped")

    def control_client(self, ranks: list[int]) -> None:
        """Set the list of clients to control and wait for their connection.

        This method blocks until all specified clients have connected
        and sent their CLIENT_HELLO message.

        Args:
            ranks: List of client ranks to control
        """
        self._controlled_ranks = set(ranks)
        logger.info("Waiting for clients with ranks: %s", ranks)

        # Wait for all controlled clients to connect
        while self._running:
            connected = {rank for rank in self._controlled_ranks if rank in self._clients}
            if connected == self._controlled_ranks:
                logger.info("All controlled clients connected: %s", list(self._controlled_ranks))
                break
            time.sleep(0.1)

    def get_client_address(self, rank: int) -> str | None:
        """Get the address of a connected client.

        Args:
            rank: The client rank

        Returns:
            The client address (IP) if connected, None otherwise
        """
        if rank in self._clients:
            address = self._clients[rank]["address"]
            # address is a tuple (ip, port), return only the IP
            return address[0] if isinstance(address, tuple) else address
        return None

    # -------------------------------------------------------------------------
    # Public Methods - Client Management
    # -------------------------------------------------------------------------

    def shutdown_client(self, ranks: list[int] | None = None) -> None:
        """Send shutdown command to clients and wait for disconnection.

        Args:
            ranks: List of client ranks to shutdown. If None, shutdown all controlled clients.
        """
        target_ranks = set(ranks) if ranks is not None else self._controlled_ranks.copy()
        logger.info("Shutting down clients with ranks: %s", list(target_ranks))

        # Send shutdown message to each target client
        for rank in target_ranks:
            if rank in self._clients:
                message = create_shutdown_client()
                self._send_message_to_client(rank, message)

        # Wait for clients to disconnect
        while self._running:
            remaining = {rank for rank in target_ranks if rank in self._clients}
            if not remaining:
                logger.info("All target clients disconnected")
                break
            time.sleep(0.1)

    def client_print(self, message_str: str, client: list[int] | None = None) -> str:
        """Send a print command to clients.

        Args:
            message_str: The string to print on clients
            client: List of client ranks to send to. If None, send to all controlled clients.

        Returns:
            A unique handle for this message to be used with wait_for_reply
        """
        target_ranks = set(client) if client is not None else self._controlled_ranks.copy()
        logger.info("Sending print message to clients %s: %s", list(target_ranks), message_str)

        # Generate unique handle for this message
        handle = str(uuid.uuid4())

        # Initialize reply tracking for this handle
        self._reply_events[handle] = {}
        self._reply_results[handle] = {}
        self._reply_payloads[handle] = {}

        for rank in target_ranks:
            self._reply_events[handle][rank] = threading.Event()
            if rank in self._clients:
                message = create_client_print(message_str, handle)
                self._send_message_to_client(rank, message)

        return handle

    def wait_for_reply(self, handle: str) -> dict[int, int]:
        """Wait for replies from clients for a specific message handle.

        A client that goes offline before replying (e.g., because its machine
        was preempted) is reported with ReplyCode.CLIENT_DISCONNECTED instead
        of blocking this call forever; when the client is one of the
        controlled ranks, every other pending reply is abandoned with
        ReplyCode.PEER_DISCONNECTED because the run cannot complete without
        the lost node.

        Args:
            handle: The message handle returned by client_print or other message methods

        Returns:
            Dictionary mapping rank to return code (0 for success, non-zero for failure)
        """
        if handle not in self._reply_events:
            logger.error("Invalid handle: %s", handle)
            return {}

        target_ranks = set(self._reply_events[handle].keys())
        logger.info("Waiting for replies from clients %s for handle %s", list(target_ranks), handle)

        results = {}
        for rank in target_ranks:
            self._wait_for_reply_event(handle, rank)
            results[rank] = self._reply_results[handle][rank]

        return results

    def wait_for_reply_with_payload(self, handle: str) -> dict[int, dict[str, Any]]:
        """Wait for replies and return both code and structured payload.

        Like wait_for_reply, a client that goes offline before replying is
        reported with ReplyCode.CLIENT_DISCONNECTED instead of blocking
        forever, and a controlled client going offline abandons every other
        pending reply with ReplyCode.PEER_DISCONNECTED.

        Args:
            handle: The message handle returned by a message method that expects a payload

        Returns:
            Dictionary mapping rank to {"code": return code, "payload": payload}
        """
        if handle not in self._reply_events:
            logger.error("Invalid handle: %s", handle)
            return {}

        target_ranks = set(self._reply_events[handle].keys())
        logger.info("Waiting for payload replies from clients %s for handle %s", list(target_ranks), handle)

        results = {}
        for rank in target_ranks:
            self._wait_for_reply_event(handle, rank)
            results[rank] = {
                "code": self._reply_results[handle][rank],
                "payload": self._reply_payloads[handle].get(rank),
            }

        return results

    # -------------------------------------------------------------------------
    # Private Methods - Reply Tracking
    # -------------------------------------------------------------------------

    def _wait_for_reply_event(self, handle: str, rank: int) -> None:
        """Wait for one rank's reply event, failing fast if the client goes offline.

        The reply event alone cannot notice that a client closed its
        connection without replying (for example because its machine was
        preempted), so the connection is polled while waiting: once the client
        is gone and its reply is still outstanding, the pending reply is
        failed so callers such as run.py exit with an error instead of
        hanging forever.

        Args:
            handle: The message handle being waited on.
            rank: The client rank expected to reply.
        """
        event = self._reply_events[handle][rank]
        # Backstop for the dispatch-vs-disconnect race that the immediate
        # failure propagation in _handle_client_connection cannot cover: a
        # dispatch method checks "rank in self._clients" first and creates the
        # reply event only afterwards, so a disconnect processed exactly inside
        # that gap fails nothing (the event does not exist yet) and the event
        # created afterwards is sent to a dead socket that never replies.
        # Polling catches such an orphaned event within one interval; while
        # the rank stays connected the poll only costs one wakeup per interval.
        while not event.wait(timeout=_REPLY_WAIT_POLL_INTERVAL_SECONDS):
            if rank not in self._clients:
                self._fail_pending_reply(handle, rank)
                return

    def _fail_pending_reply(self, handle: str, rank: int, code: int = ReplyCode.CLIENT_DISCONNECTED) -> None:
        """Fail a pending reply whose result can no longer arrive.

        Marks the reply with the given code and fires its event so
        wait_for_reply stops blocking; a reply that can never arrive (the
        client went offline, or a peer's offline dooms the run) would
        otherwise hang the waiter forever.

        Args:
            handle: The message handle of the pending request.
            rank: The client rank that was expected to reply.
            code: The ReplyCode to report; CLIENT_DISCONNECTED for the rank
                that went offline itself, PEER_DISCONNECTED for ranks whose
                replies are abandoned because a peer went offline.
        """
        events = self._reply_events.get(handle)
        if events is None:
            return
        event = events.get(rank)
        if event is None or event.is_set():
            return
        results = self._reply_results.get(handle)
        payloads = self._reply_payloads.get(handle)
        if results is None or payloads is None:
            return

        logger.error(
            "Failing pending reply from rank %s for handle %s: %s",
            rank,
            handle,
            describe_reply_code(code),
        )
        # Write the result before firing the event so the waiter reads consistent data.
        results[rank] = code
        payloads[rank] = None
        event.set()

    def _fail_pending_replies_for_disconnect(self, disconnected_rank: int) -> None:
        """Fail pending replies after a rank's connection closed.

        The disconnected rank itself is reported with CLIENT_DISCONNECTED.
        When it is one of the controlled ranks, every other pending reply is
        abandoned with PEER_DISCONNECTED as well: distributed training cannot
        complete after losing a node, the surviving clients can only report a
        doomed run after their NCCL timeout, and waiting for that adds no
        information - failing everything now lets run.py exit immediately and
        the batch runner retry without the timeout delay. A client outside
        the controlled ranks only fails its own pending replies.

        Args:
            disconnected_rank: The client rank whose connection closed.
        """
        fail_all = disconnected_rank in self._controlled_ranks
        if fail_all:
            logger.error(
                "Controlled client rank %s went offline; abandoning every pending reply because the "
                "run cannot complete without it",
                disconnected_rank,
            )
        for handle in list(self._reply_events):
            for rank in list(self._reply_events[handle]):
                if rank == disconnected_rank:
                    self._fail_pending_reply(handle, rank)
                elif fail_all:
                    self._fail_pending_reply(handle, rank, ReplyCode.PEER_DISCONNECTED)

    # -------------------------------------------------------------------------
    # Public Methods - Training Task Management
    # -------------------------------------------------------------------------

    def dispatch_train_task(
        self,
        master_addr: str,
        master_port: int,
        node_rank: int,
        log_dir_name: str,
        train_script_path: str,
        extra_args_str: str,
        extra_envs_str: str,
        msg_target_node_rank: int,
    ) -> str:
        """Dispatch a training task to a specific node."""
        logger.info("Dispatching train task to msg_target_node_rank=%s (node_rank=%s)", msg_target_node_rank, node_rank)

        # Generate unique handle for this message
        handle = str(uuid.uuid4())

        # Initialize reply tracking for this handle
        self._reply_events[handle] = {}
        self._reply_results[handle] = {}
        self._reply_payloads[handle] = {}

        if msg_target_node_rank in self._clients:
            self._reply_events[handle][msg_target_node_rank] = threading.Event()
            message = create_train_start(
                master_addr=master_addr,
                master_port=master_port,
                node_rank=node_rank,
                log_dir_name=log_dir_name,
                train_script_path=train_script_path,
                extra_args_str=extra_args_str,
                extra_envs_str=extra_envs_str,
                handle=handle,
            )
            self._send_message_to_client(msg_target_node_rank, message)
        else:
            logger.warning("Target rank %s is not connected, cannot dispatch train task", msg_target_node_rank)

        return handle

    def dispatch_train_wait(self, msg_target_node_ranks: list[int] | None = None) -> str:
        """Send TRAIN_WAIT message to clients and wait for training completion.

        This method sends a TRAIN_WAIT message to the specified clients,
        which instructs them to wait for the training process to complete
        and return the result.
        """
        msg_target_nodes = (
            set(msg_target_node_ranks) if msg_target_node_ranks is not None else self._controlled_ranks.copy()
        )
        logger.info("Sending TRAIN_WAIT to msg_target_nodse=%s", list(msg_target_nodes))

        # Generate unique handle for this message
        handle = str(uuid.uuid4())

        # Initialize reply tracking for this handle
        self._reply_events[handle] = {}
        self._reply_results[handle] = {}
        self._reply_payloads[handle] = {}

        for target in msg_target_nodes:
            if target in self._clients:
                self._reply_events[handle][target] = threading.Event()
                message = create_train_wait(handle=handle)
                self._send_message_to_client(target, message)
            else:
                logger.warning("Target rank %s is not connected, cannot send TRAIN_WAIT", target)

        return handle

    def stop_train_on_nodes(self, msg_target_node_ranks: list[int]) -> None:
        """Stop training on specified target ranks."""
        logger.info("Stopping training on msg_target_node_ranks=%s", msg_target_node_ranks)

        for target_rank in msg_target_node_ranks:
            if target_rank in self._clients:
                message = create_train_stop()
                self._send_message_to_client(target_rank, message)
            else:
                logger.warning("Target rank %s is not connected, cannot stop training", target_rank)

    # -------------------------------------------------------------------------
    # Public Methods - Mock Bootstrap Management
    # -------------------------------------------------------------------------

    def prepare_mock_bootstrap(
        self,
        msg_target_node_rank: int,
        redis_host: str,
        trace_dir: str,
        nnodes: int,
        sandbox_nodes: list[int],
        gpus_per_node: int,
    ) -> str:
        """Prepare mock NCCL bootstrap Redis state on a specific node."""
        logger.info("Preparing mock NCCL bootstrap on msg_target_node_rank=%s", msg_target_node_rank)

        handle = str(uuid.uuid4())
        self._reply_events[handle] = {}
        self._reply_results[handle] = {}
        self._reply_payloads[handle] = {}

        if msg_target_node_rank in self._clients:
            self._reply_events[handle][msg_target_node_rank] = threading.Event()
            message = create_mock_bootstrap_prepare(
                redis_host=redis_host,
                trace_dir=trace_dir,
                nnodes=nnodes,
                sandbox_nodes=sandbox_nodes,
                gpus_per_node=gpus_per_node,
                handle=handle,
            )
            self._send_message_to_client(msg_target_node_rank, message)
        else:
            logger.warning("Target rank %s is not connected, cannot prepare mock bootstrap", msg_target_node_rank)

        return handle

    # -------------------------------------------------------------------------
    # Public Methods - Assistant Management
    # -------------------------------------------------------------------------

    def start_assistants(
        self,
        msg_target_node_rank: int,
        master_addr: str,
        master_port: int,
        redis_host: str,
        log_dir_name: str,
        emulated_nodes_for_this_assistant: list[int],
        nnodes: int,
        sandbox_nodes: list[int],
        assistant_envs_str: str,
        trace_dir: str,
        graph_dir: str | None,
        graph_cache_path: str | None,
        profiler_dir: str | None,
        pp_rank_stride: int | None,
        enable_assistant_online_update: bool,
        tensor_provider: str | None = None,
        enable_mock_nccl_bootstrap: bool = False,
        assistant_nsight_system_ranks: list[int] | None = None,
        gpus_per_node: int = 8,
    ) -> str:
        """Start assistant processes on a specific node."""
        logger.info(
            "Starting assistants on msg_target_node_rank=%s, emulated_nodes_for_this_assistant=%s",
            msg_target_node_rank,
            emulated_nodes_for_this_assistant,
        )

        # Generate unique handle for this message
        handle = str(uuid.uuid4())

        # Initialize reply tracking for this handle
        self._reply_events[handle] = {}
        self._reply_results[handle] = {}
        self._reply_payloads[handle] = {}

        if msg_target_node_rank in self._clients:
            self._reply_events[handle][msg_target_node_rank] = threading.Event()
            message = create_assistant_start(
                master_addr=master_addr,
                master_port=master_port,
                redis_host=redis_host,
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
                handle=handle,
            )
            self._send_message_to_client(msg_target_node_rank, message)
        else:
            logger.warning("Target rank %s is not connected, cannot start assistants", msg_target_node_rank)

        return handle

    def stop_assistants(self, msg_target_node_ranks: int) -> None:
        """Stop assistant processes on a specific target rank."""
        logger.info("Stopping assistants on msg_target_node_ranks=%s", msg_target_node_ranks)

        if msg_target_node_ranks in self._clients:
            message = create_assistant_stop()
            self._send_message_to_client(msg_target_node_ranks, message)
        else:
            logger.warning("Target rank %s is not connected, cannot stop assistants", msg_target_node_ranks)

    # -------------------------------------------------------------------------
    # Private Methods - Connection Handling
    # -------------------------------------------------------------------------

    def _accept_connections(self) -> None:
        """Accept incoming client connections."""
        while self._running:
            try:
                client_socket, address = self._server_socket.accept()
                logger.info("Accepted connection from %s", address)

                # Start a handler thread for this client
                handler_thread = threading.Thread(
                    target=self._handle_client_connection,
                    args=(client_socket, address),
                    daemon=True,
                )
                handler_thread.start()
            except OSError:
                if self._running:
                    logger.error("Error accepting client connection")
                break

    def _handle_client_connection(self, client_socket: socket.socket, address: tuple[str, int]) -> None:
        """Handle communication with a single client.

        Args:
            client_socket: The socket connected to the client
            address: The client's address
        """
        buffer = ""
        client_rank: int | None = None

        def change_client_rank(new_rank):
            nonlocal client_rank
            client_rank = new_rank

        try:
            while self._running:
                try:
                    chunk = client_socket.recv(4096).decode("utf-8")
                    if not chunk:
                        logger.info("Connection from %s closed by client", address)
                        break

                    buffer += chunk

                    # Process complete messages (newline-delimited)
                    while "\n" in buffer:
                        message_str, buffer = buffer.split("\n", 1)
                        if message_str.strip():
                            # Directly handle the message in this thread
                            self._handle_message(message_str, client_socket, address, change_client_rank)
                except OSError as e:
                    if self._running:
                        logger.error("Error handling connection from %s: %s", address, e)
                    break
        finally:
            # Clean up: close socket and remove client from registry
            with contextlib.suppress(OSError):
                client_socket.close()
            if client_rank is not None and client_rank in self._clients:
                with self._clients_lock:
                    if client_rank in self._clients:
                        del self._clients[client_rank]
                        # The client went offline (e.g., its machine was
                        # preempted) and can never reply to pending requests.
                        # Hold the lock so a reconnecting client with the same
                        # rank cannot register in between; the helper snapshots
                        # the reply tables since the main thread may create new
                        # entries concurrently.
                        self._fail_pending_replies_for_disconnect(client_rank)

    # -------------------------------------------------------------------------
    # Private Methods - Message Handling
    # -------------------------------------------------------------------------

    def _handle_message(
        self,
        data: str,
        client_socket: socket.socket,
        address: tuple[str, int],
        client_rank_handle: Callable[[int], None],
    ) -> None:
        """Handle a single message from a client.

        Args:
            data: The raw message data
            client_socket: The socket connected to the client
            address: The client's address
            client_rank_handle: The function to call when receiving the client rank
        """
        try:
            message = decode_message(data)
            logger.info("Received message from %s: %s with data %s", address, message.type, message.data)

            if message.type == MessageType.CLIENT_HELLO:
                client_rank = self._handle_client_hello(message, client_socket, address)
                if client_rank is not None:
                    client_rank_handle(client_rank)
            elif message.type == MessageType.CLIENT_REPLY:
                self._handle_client_reply(message)
            else:
                logger.warning("Unknown message type: %s", message.type)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from %s: %s", address, data)
        except Exception as e:
            logger.error("Error handling message from %s: %s", address, e)

    def _handle_client_hello(
        self, message: Message, client_socket: socket.socket, address: tuple[str, int]
    ) -> int | None:
        """Handle CLIENT_HELLO message.

        Args:
            message: The received message
            client_socket: The socket connected to the client
            address: The client's address

        Returns:
            The client rank if registration successful, None otherwise
        """
        rank = message.data.get("rank")
        if rank is None:
            logger.error("CLIENT_HELLO missing rank from %s", address)
            return None

        with self._clients_lock:
            # Check if rank is already registered
            if rank in self._clients:
                logger.warning(
                    "Rank %s is already registered from %s, rejecting new connection from %s",
                    rank,
                    self._clients[rank]["address"],
                    address,
                )
                # First send an error message to the new client
                error_handle = str(uuid.uuid4())
                error_message = create_client_print(
                    f"Error: Rank {rank} is already registered by another client {self._clients[rank]['address']}",
                    error_handle,
                )
                try:
                    client_socket.send(encode_message(error_message))
                    logger.info("Sent error message to client %s", address)
                except OSError as e:
                    logger.error("Failed to send error message to %s: %s", address, e)
                # Then send shutdown message to gracefully terminate the client
                shutdown_message = create_shutdown_client()
                try:
                    client_socket.send(encode_message(shutdown_message))
                    logger.info("Sent shutdown message to client %s", address)
                except OSError as e:
                    logger.error("Failed to send shutdown message to %s: %s", address, e)
                return None

            self._clients[rank] = {
                "socket": client_socket,
                "address": address,
            }
        logger.info("Registered client rank %s from %s", rank, address)
        return rank

    def _handle_client_reply(self, message: Message) -> None:
        """Handle CLIENT_REPLY message.

        Args:
            message: The received message
        """
        code = message.data.get("code")
        rank = message.data.get("_rank")
        handle = message.data.get("_handle")
        msg = message.data.get("msg", "")
        payload = message.data.get("payload")

        if rank not in self._clients:
            logger.warning("CLIENT_REPLY from unknown rank %s", rank)
            return
        if handle not in self._reply_results or handle not in self._reply_events:
            logger.warning("CLIENT_REPLY for unknown handle %s", handle)
            return

        logger.info("Received reply from rank %s: code=%s, handle=%s", rank, code, handle)

        # Print a diagnostic when the reply reports a failure
        if code != ReplyCode.SUCCESS:
            logger.error("Error reply from rank %s: %s", rank, msg or describe_reply_code(code))

        # Store reply result for the corresponding handle
        self._reply_results[handle][rank] = code
        self._reply_payloads[handle][rank] = payload
        self._reply_events[handle][rank].set()

    # -------------------------------------------------------------------------
    # Private Methods - Message Sending
    # -------------------------------------------------------------------------

    def _send_message_to_client(self, rank: int, message: Message) -> None:
        """Send a message to a specific client.

        Args:
            rank: The client rank
            message: The message to send
        """
        if rank not in self._clients:
            logger.warning("Cannot send message to rank %s: not connected", rank)
            return
        client_socket = self._clients[rank]["socket"]

        try:
            client_socket.send(encode_message(message))
            logger.info("Sent message to rank %s: %s", rank, message.type)
        except OSError as e:
            logger.error("Failed to send message to rank %s: %s", rank, e)

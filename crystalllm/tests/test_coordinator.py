# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Comprehensive tests for CrystalLLM Coordinator."""

import json
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

from crystalllm.coordinator import CrystalClient, CrystalServer
from crystalllm.coordinator.protocol import (
    MIN_SIGNAL_CODE,
    Message,
    MessageType,
    ReplyCode,
    create_client_hello,
    create_client_print,
    create_client_reply,
    create_shutdown_client,
    create_train_wait,
    decode_message,
    describe_reply_code,
    encode_message,
)
from crystalllm.runner.base_runner import BaseRunner


class TestMessageProtocol(unittest.TestCase):
    """Test message protocol serialization/deserialization."""

    def test_message_roundtrip(self):
        """Test messages survive to_json/from_json round trips, including default empty data."""
        original = Message(type=MessageType.CLIENT_HELLO, data={"rank": 0})
        restored = Message.from_json(original.to_json())

        self.assertEqual(restored.type, MessageType.CLIENT_HELLO)
        self.assertEqual(restored.data, {"rank": 0})

        # Messages created without explicit data default to an empty payload.
        empty = Message.from_json(Message(type=MessageType.SHUTDOWN_CLIENT).to_json())
        self.assertEqual(empty.type, MessageType.SHUTDOWN_CLIENT)
        self.assertEqual(empty.data, {})

    def test_encode_decode_message(self):
        """Test encode_message appends the newline delimiter and decode_message accepts bytes or str."""
        encoded = encode_message(create_client_hello(rank=5))
        self.assertTrue(encoded.endswith(b"\n"))

        decoded = decode_message(encoded)
        self.assertEqual(decoded.type, MessageType.CLIENT_HELLO)
        self.assertEqual(decoded.data["rank"], 5)

        # decode_message accepts both bytes and str input.
        json_str = '{"type": "CLIENT_REPLY", "data": {"code": 0}}'
        for data in (json_str.encode("utf-8"), json_str):
            self.assertEqual(decode_message(data).type, MessageType.CLIENT_REPLY)

    def test_message_from_json_rejects_invalid_payload(self):
        """Test Message.from_json raises on invalid JSON and on a missing type field."""
        with self.assertRaises(json.JSONDecodeError):
            Message.from_json("not valid json")

        with self.assertRaises(KeyError):
            Message.from_json('{"data": {}}')


class TestReplyCode(unittest.TestCase):
    """Test ReplyCode definitions and describe_reply_code."""

    def test_protocol_codes_stay_out_of_subprocess_ranges(self):
        """Test protocol codes do not collide with exit codes or signal codes."""
        for code in ReplyCode:
            if code == ReplyCode.SUCCESS:
                self.assertEqual(code, 0)
                continue
            self.assertLess(
                code,
                MIN_SIGNAL_CODE,
                f"{code!r} must stay outside the exit code (0..255) and signal (-1..-64) ranges",
            )

    def test_reply_code_serializes_as_plain_int(self):
        """Test ReplyCode members serialize to their integer value in messages."""
        reply = create_client_reply(ReplyCode.RUNNER_QUOTA_IN_USE, 3, "h1")
        parsed = json.loads(reply.to_json())
        self.assertEqual(parsed["data"]["code"], -101)

    def test_describe_reply_code(self):
        """Test describe_reply_code explains protocol codes and raw exit codes."""
        self.assertEqual(describe_reply_code(0), "success")
        self.assertEqual(describe_reply_code(ReplyCode.INTERNAL_ERROR), "client internal error")
        self.assertEqual(describe_reply_code(ReplyCode.RUNNER_QUOTA_IN_USE), "runner quota already in use on the node")
        self.assertEqual(describe_reply_code(ReplyCode.NO_ACTIVE_RUNNER), "no active runner on the node")
        self.assertEqual(
            describe_reply_code(ReplyCode.CLIENT_DISCONNECTED),
            "client disconnected before replying (node offline or preempted)",
        )
        self.assertEqual(describe_reply_code(ReplyCode.PEER_DISCONNECTED), "abandoned because a peer node went offline")
        self.assertEqual(describe_reply_code(1), "process exited with code 1")
        self.assertEqual(describe_reply_code(137), "process exited with code 137")
        self.assertEqual(describe_reply_code(-9), "process killed by signal 9")
        self.assertEqual(describe_reply_code(-12345), "reply code -12345")


class TestClientServerIntegration(unittest.TestCase):
    """Test Client-Server integration with real sockets."""

    def setUp(self):
        """Set up server and find available port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self.host = "localhost"
        self.server = CrystalServer(host=self.host, port=self.port)

    def tearDown(self):
        """Clean up server."""
        if self.server._running:
            self.server.stop_server()

    def test_server_start_stop_lifecycle(self):
        """Test server can start and stop correctly."""
        self.server.start_server()
        self.assertTrue(self.server._running)
        self.assertIsNotNone(self.server._server_socket)

        self.server.stop_server()
        self.assertFalse(self.server._running)
        self.assertIsNone(self.server._server_socket)

    def test_server_port_in_use_raises_error(self):
        """Test server raises error when port is already in use."""
        self.server.start_server()

        server2 = CrystalServer(host=self.host, port=self.port)
        with self.assertRaises(OSError):
            server2.start_server()

    def test_client_registers_then_shutdown_removes_it(self):
        """Test a client registers via CLIENT_HELLO and shutdown_client removes it from the registry."""
        self.server.start_server()

        client = CrystalClient(host=self.host, port=self.port, rank=0, crystal_workspace=tempfile.mkdtemp())
        threading.Thread(target=client.start, daemon=True).start()

        # control_client blocks until the client has registered, so no sleep is needed.
        self.server.control_client([0])
        self.assertIn(0, self.server._clients)
        self.assertEqual(self.server.get_client_address(0), "127.0.0.1")

        # shutdown_client waits for the client to actually disconnect.
        self.server.shutdown_client([0])
        self.assertNotIn(0, self.server._clients)

    def test_control_client_blocks_until_all_connected(self):
        """Test control_client blocks until all specified clients connect."""
        self.server.start_server()

        connected_event = threading.Event()

        def control_clients():
            self.server.control_client([0, 1])
            connected_event.set()

        control_thread = threading.Thread(target=control_clients, daemon=True)
        control_thread.start()

        time.sleep(0.2)
        self.assertFalse(connected_event.is_set(), "control_client should block until clients connect")

        client0 = CrystalClient(host=self.host, port=self.port, rank=0, crystal_workspace=tempfile.mkdtemp())
        client1 = CrystalClient(host=self.host, port=self.port, rank=1, crystal_workspace=tempfile.mkdtemp())

        threading.Thread(target=client0.start, daemon=True).start()
        threading.Thread(target=client1.start, daemon=True).start()

        connected_event.wait(timeout=2.0)
        self.assertTrue(connected_event.is_set())

        client0.stop()
        client1.stop()

    def test_client_print_and_wait_for_reply(self):
        """Test client_print sends message and client replies correctly."""
        self.server.start_server()

        printed_messages = []

        def mock_print(*args, **kwargs):
            printed_messages.append(" ".join(str(a) for a in args))

        client = CrystalClient(host=self.host, port=self.port, rank=0, crystal_workspace=tempfile.mkdtemp())

        def run_client():
            with mock.patch("builtins.print", mock_print):
                client.start()

        threading.Thread(target=run_client, daemon=True).start()

        self.server.control_client([0])

        handle = self.server.client_print("Hello from server", client=[0])
        results = self.server.wait_for_reply(handle)

        self.assertEqual(results, {0: ReplyCode.SUCCESS})
        # Printing the text is the payload of the CLIENT_PRINT message, so the
        # server's text must reach the client's print verbatim; what else the
        # client prints is not this test's business.
        self.assertIn("Hello from server", printed_messages)

        client.stop()

    def test_multiple_clients_broadcast_and_targeted_messages(self):
        """Test communication with multiple clients - broadcast and targeted."""
        self.server.start_server()

        clients = []
        for rank in [0, 1, 2]:
            client = CrystalClient(host=self.host, port=self.port, rank=rank, crystal_workspace=tempfile.mkdtemp())
            clients.append(client)
            threading.Thread(target=client.start, daemon=True).start()

        self.server.control_client([0, 1, 2])

        # Test broadcast message
        handle = self.server.client_print("Broadcast message")
        results = self.server.wait_for_reply(handle)
        self.assertEqual(results, {0: ReplyCode.SUCCESS, 1: ReplyCode.SUCCESS, 2: ReplyCode.SUCCESS})

        # Test targeted message to specific client
        handle = self.server.client_print("Targeted message", client=[1])
        results = self.server.wait_for_reply(handle)
        self.assertEqual(results, {1: ReplyCode.SUCCESS})

        for client in clients:
            client.stop()


class TestClientDisconnectDuringWait(unittest.TestCase):
    """Test that pending reply waits fail fast when a client goes offline."""

    def setUp(self):
        """Set up a running server on a free port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self.host = "localhost"
        self.server = CrystalServer(host=self.host, port=self.port)
        self.server.start_server()

    def tearDown(self):
        """Clean up server."""
        if self.server._running:
            self.server.stop_server()

    def _wait_until(self, condition):
        """Block until the condition holds or a timeout elapses."""
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if condition():
                return
            time.sleep(0.05)
        self.fail("Timed out waiting for condition")

    def _connect_raw_client(self, rank):
        """Connect a raw socket client and register it with CLIENT_HELLO."""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        client_socket.sendall(encode_message(create_client_hello(rank)))
        self._wait_until(lambda: rank in self.server._clients)
        return client_socket

    def test_wait_for_reply_fails_when_client_disconnects(self):
        """Test a client going offline mid-wait fails the reply instead of blocking forever."""
        client_socket = self._connect_raw_client(0)

        handle = self.server.dispatch_train_wait([0])

        wait_results = {}
        wait_thread = threading.Thread(
            target=lambda: wait_results.update(self.server.wait_for_reply(handle)),
            daemon=True,
        )
        wait_thread.start()

        # Simulate a preempted client machine: the connection closes without any reply.
        client_socket.close()

        wait_thread.join(timeout=5.0)
        self.assertFalse(wait_thread.is_alive(), "wait_for_reply must return after the client disconnects")
        self.assertEqual(wait_results, {0: ReplyCode.CLIENT_DISCONNECTED})

        # The payload API reports the same failed reply with an empty payload.
        self.assertEqual(
            self.server.wait_for_reply_with_payload(handle),
            {0: {"code": ReplyCode.CLIENT_DISCONNECTED, "payload": None}},
        )

    def test_wait_for_reply_succeeds_when_reply_precedes_disconnect(self):
        """Test a reply received before the disconnect is not replaced by a failure."""
        client_socket = self._connect_raw_client(0)

        handle = self.server.dispatch_train_wait([0])
        client_socket.sendall(encode_message(create_client_reply(code=0, rank=0, handle=handle)))

        reply_event = self.server._reply_events[handle][0]
        self._wait_until(reply_event.is_set)
        client_socket.close()

        results = self.server.wait_for_reply(handle)
        self.assertEqual(results, {0: 0})

    def test_wait_for_reply_mixes_success_and_disconnect(self):
        """Test an uncontrolled disconnecting rank fails only its own reply."""
        # No control_client call: both ranks are uncontrolled, so the
        # disconnect only fails rank 0's own reply, not rank 1's.
        client0 = self._connect_raw_client(0)
        client1 = self._connect_raw_client(1)

        handle = self.server.client_print("ping", client=[0, 1])
        client1.sendall(encode_message(create_client_reply(code=0, rank=1, handle=handle)))

        reply_event = self.server._reply_events[handle][1]
        self._wait_until(reply_event.is_set)
        client0.close()

        results = self.server.wait_for_reply(handle)
        self.assertEqual(results, {0: ReplyCode.CLIENT_DISCONNECTED, 1: 0})

        client1.close()

    def test_controlled_rank_disconnect_abandons_all_pending_replies(self):
        """Test a controlled rank going offline abandons every pending reply."""
        client0 = self._connect_raw_client(0)
        client1 = self._connect_raw_client(1)
        self.server.control_client([0, 1])

        handle = self.server.dispatch_train_wait([0, 1])

        wait_results = {}
        wait_thread = threading.Thread(
            target=lambda: wait_results.update(self.server.wait_for_reply(handle)),
            daemon=True,
        )
        wait_thread.start()

        client0.close()

        wait_thread.join(timeout=5.0)
        self.assertFalse(wait_thread.is_alive(), "wait_for_reply must return after a controlled rank disconnects")
        self.assertEqual(
            wait_results,
            {0: ReplyCode.CLIENT_DISCONNECTED, 1: ReplyCode.PEER_DISCONNECTED},
        )

        client1.close()

    def test_controlled_disconnect_keeps_replies_that_already_arrived(self):
        """Test the peer mass failure does not overwrite replies that already arrived."""
        client0 = self._connect_raw_client(0)
        client1 = self._connect_raw_client(1)
        self.server.control_client([0, 1])

        handle = self.server.client_print("ping", client=[0, 1])
        client1.sendall(encode_message(create_client_reply(code=0, rank=1, handle=handle)))

        reply_event = self.server._reply_events[handle][1]
        self._wait_until(reply_event.is_set)
        client0.close()

        results = self.server.wait_for_reply(handle)
        self.assertEqual(results, {0: ReplyCode.CLIENT_DISCONNECTED, 1: 0})

        client1.close()

    def test_wait_for_reply_fails_for_rank_without_connection(self):
        """Test a pending reply for a rank with no connection fails via the wait poll."""
        handle = "orphan_handle"
        self.server._reply_events[handle] = {7: threading.Event()}
        self.server._reply_results[handle] = {}
        self.server._reply_payloads[handle] = {}

        wait_results = {}
        wait_thread = threading.Thread(
            target=lambda: wait_results.update(self.server.wait_for_reply(handle)),
            daemon=True,
        )
        wait_thread.start()

        wait_thread.join(timeout=5.0)
        self.assertFalse(wait_thread.is_alive(), "wait_for_reply must fail the reply of an offline rank")
        self.assertEqual(wait_results, {7: ReplyCode.CLIENT_DISCONNECTED})


class TestClientHelloHandling(unittest.TestCase):
    """Test CLIENT_HELLO registration on the server."""

    def setUp(self):
        """Set up server without starting it; _handle_client_hello is driven directly."""
        self.server = CrystalServer(host="localhost", port=0)

    def test_client_registration_stores_socket_and_address(self):
        """Test client registration stores socket and address correctly."""
        mock_socket = mock.Mock()
        address = ("192.168.1.1", 55555)

        result = self.server._handle_client_hello(create_client_hello(rank=10), mock_socket, address)

        self.assertEqual(result, 10)
        self.assertEqual(self.server._clients[10], {"socket": mock_socket, "address": address})

    def test_duplicate_rank_rejected_with_error_and_shutdown(self):
        """Test duplicate rank registration sends error, shutdown, and doesn't add to clients."""
        mock_socket1 = mock.Mock()
        mock_socket2 = mock.Mock()
        address1 = ("192.168.1.1", 55555)
        address2 = ("192.168.1.2", 55556)

        # Register first client
        result1 = self.server._handle_client_hello(create_client_hello(rank=5), mock_socket1, address1)
        self.assertEqual(result1, 5)
        self.assertEqual(self.server._clients[5]["socket"], mock_socket1)

        # Try to register second client with same rank
        with self.assertLogs("crystalllm.coordinator.server", level="WARNING") as cm:
            result2 = self.server._handle_client_hello(create_client_hello(rank=5), mock_socket2, address2)

        # Verify rejection behavior
        self.assertIsNone(result2)
        self.assertEqual(self.server._clients[5]["socket"], mock_socket1)
        self.assertTrue(any("already registered" in msg for msg in cm.output))

        # Verify both messages sent to second client
        self.assertEqual(mock_socket2.send.call_count, 2)

        # First message should be CLIENT_PRINT error
        first_call = json.loads(mock_socket2.send.call_args_list[0][0][0].decode("utf-8"))
        self.assertEqual(first_call["type"], "CLIENT_PRINT")
        self.assertIn("already registered", first_call["data"]["str"])

        # Second message should be SHUTDOWN_CLIENT
        second_call = json.loads(mock_socket2.send.call_args_list[1][0][0].decode("utf-8"))
        self.assertEqual(second_call["type"], "SHUTDOWN_CLIENT")

    def test_new_client_can_register_after_disconnect(self):
        """Test new client with same rank can register after original disconnects."""
        mock_socket1 = mock.Mock()
        mock_socket2 = mock.Mock()
        address1 = ("192.168.1.1", 55555)
        address2 = ("192.168.1.2", 55556)

        # Register first client
        self.server._handle_client_hello(create_client_hello(rank=10), mock_socket1, address1)
        self.assertIn(10, self.server._clients)

        # Simulate disconnect
        del self.server._clients[10]

        # New client with same rank should register successfully
        result = self.server._handle_client_hello(create_client_hello(rank=10), mock_socket2, address2)

        self.assertEqual(result, 10)
        self.assertEqual(self.server._clients[10]["socket"], mock_socket2)


class TestErrorHandling(unittest.TestCase):
    """Test server error handling and edge cases."""

    def setUp(self):
        """Set up server without starting it; these tests drive handlers directly."""
        self.server = CrystalServer(host="localhost", port=0)

    def test_wait_for_reply_invalid_handle_returns_empty(self):
        """Test wait_for_reply with invalid handle returns empty dict."""
        results = self.server.wait_for_reply("nonexistent_handle")
        self.assertEqual(results, {})

    def test_send_to_disconnected_client_logs_warning(self):
        """Test sending to disconnected client logs warning."""
        with self.assertLogs("crystalllm.coordinator.server", level="WARNING") as cm:
            self.server._send_message_to_client(999, create_shutdown_client())

        self.assertIn("not connected", cm.output[0])

    def test_client_reply_with_unknown_rank_or_handle_is_ignored(self):
        """Test CLIENT_REPLY from an unknown rank or for an unknown handle logs a warning."""
        # Unknown rank
        message = create_client_reply(code=0, rank=999, handle="test")
        with self.assertLogs("crystalllm.coordinator.server", level="WARNING") as cm:
            self.server._handle_client_reply(message)
        self.assertIn("unknown rank", cm.output[0])

        # Known rank but unknown handle
        self.server._clients[0] = {"socket": None, "address": ("localhost", 12345)}
        message = create_client_reply(code=0, rank=0, handle="unknown_handle")
        with self.assertLogs("crystalllm.coordinator.server", level="WARNING") as cm:
            self.server._handle_client_reply(message)
        self.assertIn("unknown handle", cm.output[0])

    def test_client_hello_missing_rank_logs_error(self):
        """Test handling CLIENT_HELLO with missing rank logs error."""
        message = Message(type=MessageType.CLIENT_HELLO, data={})
        with self.assertLogs("crystalllm.coordinator.server", level="ERROR") as cm:
            result = self.server._handle_client_hello(message, None, ("localhost", 12345))

        self.assertIsNone(result)
        self.assertIn("missing rank", cm.output[0])

    def test_decode_invalid_json_logs_error(self):
        """Test handling invalid JSON in message logs error."""
        with self.assertLogs("crystalllm.coordinator.server", level="ERROR") as cm:
            self.server._handle_message("invalid json", None, ("localhost", 12345), lambda r: None)

        self.assertTrue(any("Failed to decode" in msg for msg in cm.output))


class TestClientMessageHandling(unittest.TestCase):
    """Test CrystalClient message handling."""

    def test_client_handles_shutdown_message(self):
        """Test client handles SHUTDOWN_CLIENT message."""
        client = CrystalClient(host="localhost", port=12345, rank=0, crystal_workspace=tempfile.mkdtemp())
        client._running = True

        message = create_shutdown_client()
        client._handle_shutdown(message)

        self.assertFalse(client._running)

    def test_client_handles_print_message_and_replies(self):
        """Test client handles CLIENT_PRINT message and sends reply."""
        client = CrystalClient(host="localhost", port=12345, rank=0, crystal_workspace=tempfile.mkdtemp())

        mock_socket = mock.Mock()
        client._socket = mock_socket

        message = create_client_print(message_str="Print this", handle="h123")

        with mock.patch("builtins.print") as mock_print:
            client._handle_print(message)
        # The whole message is printed as one positional argument, so print
        # options such as flush= may be added freely without breaking this.
        self.assertIn(("Print this",), [call.args for call in mock_print.call_args_list])

        mock_socket.send.assert_called_once()
        sent_data = json.loads(mock_socket.send.call_args[0][0].decode("utf-8"))
        self.assertEqual(sent_data["type"], "CLIENT_REPLY")
        self.assertEqual(sent_data["data"]["code"], 0)
        self.assertEqual(sent_data["data"]["_rank"], 0)
        self.assertEqual(sent_data["data"]["_handle"], "h123")

    def test_client_handles_unknown_message_type(self):
        """Test client handles unknown message type gracefully."""
        client = CrystalClient(host="localhost", port=12345, rank=0, crystal_workspace=tempfile.mkdtemp())

        # CLIENT_HELLO is a server-side message, so the client treats it as unknown.
        message = Message(type=MessageType.CLIENT_HELLO, data={})

        with self.assertLogs("crystalllm.coordinator.client", level="WARNING") as cm:
            client._handle_message(message.to_json())

        self.assertIn("Unknown message type", cm.output[0])

    def test_client_handles_message_decode_error(self):
        """Test client handles message decode error gracefully."""
        client = CrystalClient(host="localhost", port=12345, rank=0, crystal_workspace=tempfile.mkdtemp())

        with self.assertLogs("crystalllm.coordinator.client", level="ERROR") as cm:
            client._handle_message("invalid json")

        self.assertTrue(any("Error handling message" in msg for msg in cm.output))

    def test_send_message_when_not_connected_logs_error(self):
        """Test send_message when not connected logs error."""
        client = CrystalClient(host="localhost", port=12345, rank=0, crystal_workspace=tempfile.mkdtemp())
        client._socket = None

        with self.assertLogs("crystalllm.coordinator.client", level="ERROR") as cm:
            client._send_message(create_client_hello(0))

        self.assertIn("not connected", cm.output[0])

    def test_train_wait_without_runner_reports_no_active_runner(self):
        """Test TRAIN_WAIT without an active runner replies NO_ACTIVE_RUNNER."""
        client = CrystalClient(host="localhost", port=12345, rank=0, crystal_workspace=tempfile.mkdtemp())

        mock_socket = mock.Mock()
        client._socket = mock_socket

        message = create_train_wait(handle="wait_handle")
        client._handle_train_wait(message)

        sent_data = json.loads(mock_socket.send.call_args[0][0].decode("utf-8"))
        self.assertEqual(sent_data["type"], "CLIENT_REPLY")
        self.assertEqual(sent_data["data"]["code"], ReplyCode.NO_ACTIVE_RUNNER)
        self.assertEqual(sent_data["data"]["_handle"], "wait_handle")
        self.assertIn("No active TrainRunner", sent_data["data"]["msg"])


class TestBaseRunnerWait(unittest.TestCase):
    """Test BaseRunner.wait behavior that feeds CLIENT_REPLY codes."""

    class _DummyRunner(BaseRunner):
        """Concrete runner that launches no process."""

        def execute(self) -> None:
            """Launch nothing."""

    def test_wait_without_processes_still_invokes_callback(self):
        """Test wait() calls back with None when there is no process to wait for."""
        runner = self._DummyRunner()
        try:
            received = []
            return_code = runner.wait(received.append)
            self.assertIsNone(return_code)
            self.assertEqual(received, [None])
        finally:
            runner.kill()

    def test_wait_reports_subprocess_exit_code_verbatim(self):
        """Test wait() passes the subprocess exit code through to the callback."""
        runner = self._DummyRunner()
        try:
            runner._add_process(subprocess.Popen(["bash", "-c", "exit 7"]))
            received = []
            return_code = runner.wait(received.append)
            self.assertEqual(return_code, 7)
            self.assertEqual(received, [7])
        finally:
            runner.kill()

    def test_wait_reports_signal_kill_as_negative_code(self):
        """Test wait() reports a killed subprocess with the negative signal number."""
        runner = self._DummyRunner()
        try:
            runner._add_process(subprocess.Popen(["bash", "-c", "kill -9 $$"]))
            return_code = runner.wait()
            self.assertEqual(return_code, -9)
        finally:
            runner.kill()


if __name__ == "__main__":
    unittest.main()

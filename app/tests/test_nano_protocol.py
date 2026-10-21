import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nano_client import NanoClient, TriggerError


class Transport:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.pending = bytearray()
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)
        self.pending.extend(next(self.replies, b""))
        return len(data)

    def read(self, size):
        result = bytes(self.pending[:size])
        del self.pending[:size]
        return result

    def close(self):
        self.closed = True


class ClientTests(unittest.TestCase):
    def client(self, *replies):
        transport = Transport([b"READY DELSYS_SYNC 2\n", b"STATUS IDLE 0\n", *replies])
        client = NanoClient(transport, timeout=0.002)
        client.handshake()
        return client, transport

    def test_handshake_never_triggers(self):
        client, transport = self.client()
        self.assertTrue(client.ready)
        self.assertEqual(transport.writes, [b"HELLO\n", b"STATUS\n"])

    def test_start_stop_share_connection_and_increase_id(self):
        client, transport = self.client(b"ACK 1 START FIRED 2000\n", b"ACK 2 STOP FIRED 3000\n")
        start = client.trigger("START")
        stop = client.trigger("STOP")
        self.assertEqual(start["mcu_rise_us"], 2000)
        self.assertEqual(stop["event_id"], 2)
        self.assertLessEqual(start["host_write_begin_monotonic_ns"], start["host_reply_monotonic_ns"])
        client.close()
        self.assertTrue(transport.closed)
        self.assertEqual(transport.writes[-1], b"STOP 2\n")

    def test_lost_ack_never_retries_and_locks_connection(self):
        client, transport = self.client(b"")
        with self.assertRaises(TriggerError):
            client.trigger("START")
        with self.assertRaises(TriggerError):
            client.trigger("START")
        self.assertEqual(transport.writes.count(b"START 1\n"), 1)
        self.assertFalse(client.ready)

    def test_wrong_or_duplicate_ack_is_not_success(self):
        for reply in (b"ACK 2 START FIRED 1\n", b"ACK 1 STOP FIRED 1\n",
                      b"ACK 1 START DUP 1\n", b"ERR BUSY\n", b"READY DELSYS_SYNC 2\n"):
            with self.subTest(reply=reply):
                client, _ = self.client(reply)
                with self.assertRaises(TriggerError):
                    client.trigger("START")
                self.assertFalse(client.ready)

    def test_resume_uses_firmware_high_water_mark(self):
        transport = Transport([b"READY DELSYS_SYNC 2\n", b"STATUS IDLE 17\n",
                               b"ACK 18 START FIRED 4\n"])
        client = NanoClient(transport)
        client.handshake()
        self.assertEqual(client.trigger("START")["event_id"], 18)

    def test_status_detects_counter_reset(self):
        client, _ = self.client(b"ACK 1 START FIRED 4\n", b"STATUS IDLE 0\n")
        client.trigger("START")
        with self.assertRaises(TriggerError):
            client.status()

    def test_invalid_action_never_writes(self):
        client, transport = self.client()
        with self.assertRaises(ValueError):
            client.trigger("START\nSTOP 1")
        self.assertEqual(len(transport.writes), 2)

    def test_busy_handshake_refused(self):
        transport = Transport([b"READY DELSYS_SYNC 2\n", b"STATUS PULSE 1\n"])
        client = NanoClient(transport)
        with self.assertRaises(TriggerError):
            client.handshake()
        self.assertFalse(client.ready)

    def test_corrupted_or_unterminated_reply_fails_closed(self):
        for reply in (b"\xff\n", b"x" * 129, b"ACK 1 START FIRED 123"):
            with self.subTest(reply=reply):
                client, transport = self.client(reply)
                with self.assertRaises(TriggerError):
                    client.trigger("START")
                self.assertFalse(client.ready)
                self.assertEqual(transport.writes.count(b"START 1\n"), 1)

    def test_partial_write_fails_closed(self):
        client, transport = self.client()
        transport.write = lambda data: len(data) - 1
        with self.assertRaises(TriggerError):
            client.trigger("START")
        self.assertFalse(client.ready)


if __name__ == "__main__":
    unittest.main()

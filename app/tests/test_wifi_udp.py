"""UDP protocol integration against a local HTTP Shield and real UDP sockets."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

import numpy as np

from board_manager import CytonManager
from metrics import PacketLossTracker
from wifi_udp import WiFiUDPBoard


def frame(counter, values=None):
    values = values or [1, -1, 8388607, -8388608, 100, -100, 0, 7]
    return b'\xa0' + bytes([counter]) + b''.join(
        value.to_bytes(3, 'big', signed=True) for value in values
    ) + bytes(6) + b'\xc0'


class Shield:
    def __init__(self, occupied=False, rate=1000):
        self.requests = []
        self.destination = None
        self.occupied = occupied
        self.rate = rate
        self.streams = 0
        self.gains = [6] * 8
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.respond()

            def do_POST(self):
                self.respond()

            def do_DELETE(self):
                self.respond()

            def respond(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                state.requests.append((self.command, self.path, body))
                response = '{}'
                if self.path == '/board':
                    response = json.dumps({'board_type': 'cyton', 'board_connected': True})
                elif self.path == '/tcp':
                    response = json.dumps({'connected': state.occupied})
                elif self.path == '/udp' and self.command == 'POST':
                    state.destination = (body['ip'], body['port'])
                elif self.path == '/command':
                    command = body['command']
                    response = 'Success'
                    if command == '~4':
                        response = f'Success: Sample rate is {state.rate}Hz'
                    elif command.startswith('~'):
                        response = 'Failure: invalid sample value'
                    elif command == '?':
                        response = 'Board ADS Registers\n' + '\n'.join(
                            f'CH{ch + 1}SET, {ch + 5:02X}, {gain * 16 + 8:02X}'
                            for ch, gain in enumerate(state.gains)
                        ) + '\nBIAS_SENSP, 0D, FF\nBIAS_SENSN, 0E, FF\n$$$'
                    elif command.startswith('x'):
                        state.gains[int(command[1]) - 1] = int(command[3])
                elif self.path == '/stream/start':
                    state.streams += 1
                    # Include counter rollover, a missing sample, duplicate and
                    # reordering, just as a UDP network can produce.
                    state.sender.sendto(b''.join(frame(i) for i in (254, 255, 0, 2, 2, 1, 3)), state.destination)
                payload = response.encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def board(self):
        board = WiFiUDPBoard('127.0.0.1', '127.0.0.1', 0)
        original = board._opener.open

        def local_open(request, **kwargs):
            request.full_url = request.full_url.replace(
                'http://127.0.0.1/', f'http://127.0.0.1:{self.server.server_port}/'
            )
            return original(request, **kwargs)

        board._opener.open = local_open
        return board

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.sender.close()


class UDPTests(unittest.TestCase):
    def setUp(self):
        self.shield = Shield()
        self.board = self.shield.board()

    def tearDown(self):
        self.board.release_session()
        self.shield.close()

    def wait_frames(self, count):
        deadline = time.monotonic() + 2
        while self.board.receiver_status['frames'] < count and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(self.board.receiver_status['frames'], count)

    def test_http_setup_udp_receive_counters_and_restart(self):
        self.board.prepare_session()
        self.board.start_stream()
        self.wait_frames(7)
        current = self.board.get_current_board_data(3)
        self.assertEqual(current.shape, (24, 3))
        data = self.board.get_board_data()
        np.testing.assert_array_equal(data[0], [254, 255, 0, 2, 2, 1, 3])
        self.assertEqual(self.board.get_board_data().shape, (24, 0))
        self.assertTrue(np.all(np.diff(data[22]) > 0))
        tracker = PacketLossTracker(1000)
        tracker.update(data[0], data[22])
        self.assertEqual(tracker._duplicate, 1)
        self.assertEqual(tracker._out_of_order, 1)
        self.assertEqual(tracker._lost, 2)
        setup = next(body for method, path, body in self.shield.requests if (method, path) == ('POST', '/udp'))
        self.assertEqual(setup['output'], 'raw')
        self.assertFalse(setup['redundancy'])
        self.assertIn(('POST', '/command', {'command': '~4'}), self.shield.requests)
        self.assertFalse(any(body.get('command') == '~?' for _, _, body in self.shield.requests))
        self.assertFalse(any(path == '/tcp' and method != 'GET' for method, path, _ in self.shield.requests))
        self.board.stop_stream()
        self.board.start_stream()
        self.wait_frames(7)
        self.assertEqual(self.board.get_board_data().shape, (24, 7))
        self.board.release_session()
        self.assertEqual(self.shield.requests[-1][:2], ('DELETE', '/udp'))
        self.assertIsNone(self.board._socket)

    def test_signed_24bit_and_channel_gain_and_impedance_restore(self):
        self.board.prepare_session()
        expected = 4.5e6 / ((2**23 - 1) * 24)
        row = self.board._decode_datagram(frame(1), time.time())[0]
        self.assertAlmostEqual(row[1], expected)
        self.assertAlmostEqual(row[2], -expected)
        self.assertAlmostEqual(row[3], 4.5e6 / 24)
        self.assertAlmostEqual(row[4], -8388608 * expected)
        self.board.config_board('x1000010Xz101Z')
        row = self.board._decode_datagram(frame(2), time.time())[0]
        self.assertAlmostEqual(row[1], expected * 24)
        self.assertAlmostEqual(row[2], -expected)
        self.board.config_board('x1060110Xz100Z')
        row = self.board._decode_datagram(frame(3), time.time())[0]
        self.assertAlmostEqual(row[1], expected)

    def test_malformed_datagram_does_not_splice_into_next(self):
        self.assertEqual(self.board._decode_datagram(frame(1)[:20], time.time()), [])
        rows = self.board._decode_datagram(frame(1)[20:] + frame(2), time.time())
        self.assertEqual([row[0] for row in rows], [2])
        self.assertEqual(self.board.receiver_status['malformed_bytes'], 33)

    def test_refuses_other_tcp_client_without_mutation(self):
        self.shield.occupied = True
        with self.assertRaisesRegex(RuntimeError, 'TCP'):
            self.board.prepare_session()
        self.assertTrue(all(method == 'GET' for method, _, _ in self.shield.requests))

    def test_rejects_unverified_sample_rate_and_cleans_socket(self):
        self.shield.rate = 250
        with self.assertRaisesRegex(RuntimeError, '1000'):
            self.board.prepare_session()
        self.assertIsNone(self.board._socket)
        self.assertEqual(self.shield.requests[-1][:2], ('DELETE', '/udp'))

    def test_manager_hardware_configuration_updates_udp_conversion(self):
        manager = CytonManager(wifi_ip='127.0.0.1', wifi_port=0)
        with patch.object(manager, 'configure_wifi_address'), patch.object(
            manager, '_get_local_ip_for_wifi', return_value='127.0.0.1'
        ), patch.object(manager, '_probe_wifi_board', return_value='{}'), patch(
            'board_manager.WiFiUDPBoard', return_value=self.board
        ):
            self.assertTrue(manager.connect())
        try:
            manager.set_channel_settings([
                {'channel': ch, 'gain': 3, 'bias_on': True, 'srb2_on': True}
                for ch in range(1, 9)
            ])
            # Exercise the manager's existing HTTP path and gain observer.
            with patch('board_manager.urllib.request.build_opener', return_value=self.board._opener):
                self.assertEqual(manager.apply_and_verify_settings()['state'], 'verified')
            self.assertEqual(self.board._gains, [6] * 8)
            self.assertTrue(manager.start_stream(settings_prepared=True))
            self.wait_frames(7)
            data = manager.get_new_data()
            self.assertEqual(data.shape, (8, 7))
            self.assertAlmostEqual(float(data[0, 0]), 4.5e6 / ((2**23 - 1) * 6), places=6)
            self.assertTrue(manager.data_stream_verified)
        finally:
            manager.stop()


if __name__ == '__main__':
    unittest.main()

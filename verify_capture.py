"""Replay a previous diagnostic UDP recording without contacting hardware."""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'app'))
from wifi_udp import WiFiUDPBoard

session = Path(sys.argv[1])
raw = (session / 'raw.bin').read_bytes()
board = WiFiUDPBoard('127.0.0.1', '127.0.0.1', 0)
rows = []
start = time.perf_counter()
with (session / 'chunks.csv').open(encoding='utf-8', newline='') as stream:
    for chunk in csv.DictReader(stream):
        offset, length = int(chunk['offset']), int(chunk['length'])
        rows.extend(board._decode_datagram(raw[offset:offset + length], float(chunk['elapsed_s'])))
seconds = time.perf_counter() - start
data = np.stack(rows, axis=1)
with (session / 'samples.csv').open(encoding='utf-8', newline='') as stream:
    expected = list(csv.DictReader(stream))
np.testing.assert_array_equal(data[0], [int(row['sample_number']) for row in expected])
for channel in range(8):
    counts = np.array([int(row[f'ch{channel + 1}_counts']) for row in expected])
    np.testing.assert_allclose(data[channel + 1], counts * 4.5e6 / ((2**23 - 1) * 24))
assert board.receiver_status['malformed_bytes'] == 0
report = {
    'source_session': str(session), 'source_recording_rate_hz': 250,
    'payload_replay_only': True, 'frames': len(rows),
    'all_counters_match': True, 'all_8_channel_microvolt_values_match': True,
    'malformed_bytes': board.receiver_status['malformed_bytes'],
    'decode_seconds': seconds,
}
(root / 'capture_replay_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))

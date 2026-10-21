"""Persistent Nano connection. Call from a worker, never the GUI thread.

Host timestamps describe serial operations, not measured Delsys sample times.
No automatic trigger retries or reconnects are performed.
"""
import time


class TriggerError(RuntimeError):
    pass


class NanoClient:
    def __init__(self, transport, timeout=1.0):
        self.transport = transport
        self.timeout = timeout
        self.next_id = 1
        self.ready = False

    @classmethod
    def connect(cls, port):
        import serial
        transport = serial.Serial(port, 115200, timeout=0.05, write_timeout=1.0)
        client = cls(transport)
        try:
            # Nano USB DTR commonly resets the MCU; wait before the handshake.
            time.sleep(2.5)
            transport.reset_input_buffer()
            client.handshake()
        except Exception:
            transport.close()
            raise
        return client

    def _exchange(self, command):
        event = {
            "command": command,
            "host_write_begin_unix_ns": time.time_ns(),
            "host_write_begin_monotonic_ns": time.perf_counter_ns(),
        }
        payload = (command + "\n").encode("ascii")
        try:
            written = self.transport.write(payload)
            if written != len(payload):
                raise TriggerError("Incomplete serial write; execution is unknown")
            event["host_write_end_monotonic_ns"] = time.perf_counter_ns()
            deadline = time.monotonic() + self.timeout
            response = bytearray()
            while time.monotonic() < deadline:
                chunk = self.transport.read(1)
                if chunk == b"\n":
                    event["host_reply_unix_ns"] = time.time_ns()
                    event["host_reply_monotonic_ns"] = time.perf_counter_ns()
                    try:
                        event["response"] = response.decode("ascii").rstrip("\r")
                    except UnicodeDecodeError as exc:
                        raise TriggerError("Non-ASCII serial response; execution is unknown") from exc
                    return event
                response.extend(chunk)
                if len(response) > 128:
                    raise TriggerError("Response exceeds protocol limit")
            raise TriggerError("Reply timeout; execution is unknown. Do not retry automatically")
        except Exception as exc:
            self.ready = False
            event["host_error_unix_ns"] = time.time_ns()
            event["host_error_monotonic_ns"] = time.perf_counter_ns()
            exc.event = event
            raise

    def handshake(self):
        self.ready = False
        event = self._exchange("HELLO")
        if event["response"] != "READY DELSYS_SYNC 2":
            raise TriggerError("Unexpected firmware: " + event["response"])
        status = self._exchange("STATUS")["response"].split()
        if (len(status) != 3 or status[:2] != ["STATUS", "IDLE"]
                or not status[2].isascii() or not status[2].isdigit()):
            raise TriggerError("Nano is not idle or STATUS is invalid")
        last_id = int(status[2])
        if not 0 <= last_id < 0xFFFFFFFF:
            raise TriggerError("Event counter exhausted or invalid")
        self.next_id = last_id + 1
        self.ready = True

    def status(self):
        if not self.ready:
            raise TriggerError("Nano connection is not ready")
        event = self._exchange("STATUS")
        parts = event["response"].split()
        if (len(parts) != 3 or parts[0] != "STATUS"
                or parts[1] not in ("IDLE", "PULSE")
                or not parts[2].isascii() or not parts[2].isdigit()
                or int(parts[2]) != self.next_id - 1):
            self.ready = False
            raise TriggerError("Unexpected status or device reset: " + event["response"])
        return event

    def trigger(self, action):
        if action not in ("START", "STOP"):
            raise ValueError("Action must be START or STOP")
        if not self.ready:
            raise TriggerError("Nano connection is not ready")
        if self.next_id > 0xFFFFFFFF:
            raise TriggerError("Event counter exhausted")
        event_id = self.next_id
        self.next_id += 1  # Never reuse an ID after an ambiguous write.
        event = self._exchange(f"{action} {event_id}")
        parts = event["response"].split()
        if (len(parts) != 5 or parts[:3] != ["ACK", str(event_id), action]
                or parts[3] != "FIRED" or not parts[4].isascii()
                or not parts[4].isdigit() or int(parts[4]) > 0xFFFFFFFF):
            self.ready = False
            error = TriggerError("Trigger not confirmed: " + event["response"])
            error.event = event
            raise error
        event.update(event_id=event_id, action=action,
                     mcu_rise_us=int(parts[4]), status="firmware_confirmed")
        return event

    def close(self):
        # Closing is never interpreted as a Delsys STOP request.
        self.ready = False
        self.transport.close()

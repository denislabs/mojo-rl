"""Trait-based training metrics logger with pluggable backends.

Logger trait defines the interface. Concrete implementations:
  - NoOpLogger: does nothing (zero overhead, default)
  - CsvLogger: appends CSV rows to a local file
  - RemoteLogger: POSTs JSON batches to an HTTP server
  - CompositeLogger[A, B]: fans out to two loggers

Collection is pure Mojo with near-zero overhead, and so is the remote
backend: it serialises with `mojo_rl/io/json.mojo` and hands the POST to
`mojo_rl/io/http_sink.mojo`, which queues it for a background thread.

⚠ THE NETWORK IS NO LONGER ON THE TRAINING THREAD. `flush` used to POST
synchronously; against a dashboard answering in 100 ms that cost **629.7 ms**
for eleven batches, versus **0.003 ms** to queue the same eleven. A run whose
dashboard is DOWN now pays 4.6 ms for 2000 `log_scalar` calls and closes in
under 2 ms, where before it paid a connection attempt per flush.

⚠ THIS CALL IS WHY A TRAINING BINARY USED TO NEED A CPython AT ALL. `flush`
went through Python `urllib`, so every run — GPU training included — had to
find `libpython3.13`, which is what `pixi.toml`'s activation block pins and
why it names `RemoteLogger.flush` when it explains the pin. Nothing in the
training path imports Python now.

⚠ REQUIRES THE HTTP SHIM: `pixi run build-http`. A missing one is reported
once and metrics are dropped from there on — a dashboard that cannot be
reached must never take the training run with it.

⚠ METRICS ARE DROPPABLE, AND DROPS ARE COUNTED. The queue is bounded; if the
dashboard falls behind the run, batches are refused rather than stalling
training. `close()` prints the tally (`sink_report()`), and a caller that
suppresses that output is running a silently lossy logger.

Usage:
    # CSV only
    var logger = CsvLogger("logs/run_001.csv")

    # Remote only
    var logger = RemoteLogger(
        server_url="http://host:3000/api",
        run_name="ppo_halfcheetah_v3",
    )

    # Both (fan-out)
    var logger = CompositeLogger(
        CsvLogger("logs/run_001.csv"),
        RemoteLogger(server_url="http://host:3000/api"),
    )

    logger.set_config("algorithm", "PPO")
    logger.log_scalar("reward", avg_reward, step)
    logger.close()
"""

from std.time import perf_counter_ns
from std.math import isnan, isinf


from mojo_rl.io.http_sink import HttpPostSink
from mojo_rl.io.json import JsonWriter

# =============================================================================
# MetricEntry — single buffered data point
# =============================================================================


struct MetricEntry(Copyable, Movable):
    """A single scalar metric data point."""

    var step: Int
    var wall_time_ms: Float64
    var name: String
    var value: Float64

    def __init__(
        out self,
        step: Int,
        wall_time_ms: Float64,
        name: String,
        value: Float64,
    ):
        self.step = step
        self.wall_time_ms = wall_time_ms
        self.name = name
        self.value = value

    def __init__(out self, *, copy: Self):
        self.step = copy.step
        self.wall_time_ms = copy.wall_time_ms
        self.name = copy.name
        self.value = copy.value

    def __init__(out self, *, deinit move: Self):
        self.step = move.step
        self.wall_time_ms = move.wall_time_ms
        self.name = move.name^
        self.value = move.value


# =============================================================================
# Logger Trait
# =============================================================================


trait Logger(Copyable, Deinitable, Movable):
    """Interface for training metrics loggers.

    All deep RL training loops and agent structs are parameterized on
    `L: Logger = NoOpLogger`.  When L = NoOpLogger every method is a no-op
    and `is_active()` returns False, giving zero overhead identical to the
    old null-pointer pattern.
    """

    comptime ENABLED: Bool = True

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        ...

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        ...

    def flush(mut self) raises:
        ...

    def close(mut self) raises:
        ...

    def set_config(mut self, key: String, value: String):
        ...

    def is_active(self) -> Bool:
        ...


# =============================================================================
# NoOpLogger — zero-overhead default
# =============================================================================


struct NoOpLogger(Logger):
    """Logger that does nothing. Default for all training loops and agents."""

    comptime ENABLED: Bool = False

    def __init__(out self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        pass

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        pass

    def flush(mut self) raises:
        pass

    def close(mut self) raises:
        pass

    def set_config(mut self, key: String, value: String):
        pass

    def is_active(self) -> Bool:
        return False


# =============================================================================
# CsvLogger — local CSV file backend
# =============================================================================


struct CsvLogger(Logger):
    """Buffered CSV file logger.

    Accumulates MetricEntry objects and appends them to a CSV file when
    the buffer reaches `buffer_size` or on flush()/close().

    CSV format: step,wall_time_ms,name,value
    """

    var file_path: String
    var entries: List[MetricEntry]
    var buffer_size: Int
    var _start_ns: Int
    var _file_header_written: Bool
    var _total_logged: Int

    def __init__(
        out self,
        file_path: String,
        buffer_size: Int = 200,
    ):
        self.file_path = file_path
        self.entries = List[MetricEntry]()
        self.buffer_size = buffer_size
        self._start_ns = perf_counter_ns()
        self._file_header_written = False
        self._total_logged = 0

    def __init__(out self, *, deinit move: Self):
        self.file_path = move.file_path^
        self.entries = move.entries^
        self.buffer_size = move.buffer_size
        self._start_ns = move._start_ns
        self._file_header_written = move._file_header_written
        self._total_logged = move._total_logged

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        if isnan(value) or isinf(value):
            return
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        self.entries.append(MetricEntry(step, wall_time_ms, name, value))
        self._total_logged += 1
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        var n = min(len(names), len(values))
        for i in range(n):
            if isnan(values[i]) or isinf(values[i]):
                continue
            self.entries.append(
                MetricEntry(step, wall_time_ms, names[i], values[i])
            )
        self._total_logged += n
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def flush(mut self) raises:
        if len(self.entries) == 0:
            return
        var content = String("")
        if not self._file_header_written:
            content += "step,wall_time_ms,name,value\n"
            self._file_header_written = True
        for i in range(len(self.entries)):
            var e = self.entries[i].copy()
            content += (
                String(e.step)
                + ","
                + String(e.wall_time_ms)
                + ","
                + e.name
                + ","
                + String(e.value)
                + "\n"
            )
        with open(self.file_path, "a") as f:
            f.write(content)
        self.entries.clear()

    def close(mut self) raises:
        self.flush()

    def set_config(mut self, key: String, value: String):
        pass

    def is_active(self) -> Bool:
        return True

    def total_logged(self) -> Int:
        return self._total_logged

    def pending(self) -> Int:
        return len(self.entries)


# =============================================================================
# RemoteLogger — HTTP POST backend
# =============================================================================


struct RemoteLogger(Logger):
    """Buffered HTTP logger that POSTs JSON to a dashboard server.

    Sends metrics as JSON batches to `server_url/ingest` and registers
    the run at `server_url/runs` on first flush.

    ⚠ THE POST HAPPENS ON ANOTHER THREAD. `flush` serialises the batch and
    queues it; `mojo_rl/io/http_sink.mojo` owns the client and the connection.
    One client for the run either way — a client per flush would pay a full
    TLS handshake every `buffer_size` metrics — but now none of it, handshake
    included, is on the training thread.

    ⚠ ORDER IS PRESERVED, WHICH THE `/runs` REGISTRATION DEPENDS ON. One ring
    and one worker means the registration queued by the first `flush` is sent
    before the `/ingest` batch behind it.

    ⚠ `close()` IS NOT OPTIONAL. It drains the queue and joins the worker;
    without it, whatever is still queued at process exit is lost.
    """

    var run_id: String
    var run_name: String
    var server_url: String
    var api_key: String
    var entries: List[MetricEntry]
    var buffer_size: Int
    var _start_ns: Int
    var _config_keys: List[String]
    var _config_vals: List[String]
    var _run_registered: Bool
    var _total_logged: Int
    var _sink: Optional[HttpPostSink]
    """The POST queue and its background thread. Built lazily ON THE FIRST
    PAYLOAD: constructing it spawns a thread, and a `RemoteLogger` with no
    `server_url` — which is the default in several drivers — must stay inert.

    ⚠ COPIES SHARE ONE SINK, hence one thread, one queue and one connection.
    That is the right meaning (two copies of a logger are one run) and it is
    also forced: `Logger` is `Copyable`, `CompositeLogger` copies its halves,
    and a libcurl easy handle may not be shared across threads."""
    var _reported: Bool
    """Whether a transport problem has been printed. Once per run, not once
    per flush."""

    def __init__(
        out self,
        server_url: String,
        run_name: String = "",
        run_id: String = "",
        buffer_size: Int = 200,
        api_key: String = "",
    ):
        self._start_ns = perf_counter_ns()
        if run_id.byte_length() > 0:
            self.run_id = run_id
        else:
            self.run_id = "run_" + String(self._start_ns)
        self.run_name = run_name if run_name.byte_length() > 0 else self.run_id
        self.server_url = server_url
        self.api_key = api_key
        self.entries = List[MetricEntry]()
        self.buffer_size = buffer_size
        self._config_keys = List[String]()
        self._config_vals = List[String]()
        self._run_registered = False
        self._total_logged = 0
        self._sink = None
        self._reported = False

    def __init__(out self, *, deinit move: Self):
        self.run_id = move.run_id^
        self.run_name = move.run_name^
        self.server_url = move.server_url^
        self.api_key = move.api_key^
        self.entries = move.entries^
        self.buffer_size = move.buffer_size
        self._start_ns = move._start_ns
        self._config_keys = move._config_keys^
        self._config_vals = move._config_vals^
        self._run_registered = move._run_registered
        self._total_logged = move._total_logged
        self._sink = move._sink^
        self._reported = move._reported

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        if self.server_url.byte_length() == 0:
            return
        if isnan(value) or isinf(value):
            return
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        self.entries.append(MetricEntry(step, wall_time_ms, name, value))
        self._total_logged += 1
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        if self.server_url.byte_length() == 0:
            return
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0
        var n = min(len(names), len(values))
        for i in range(n):
            if isnan(values[i]) or isinf(values[i]):
                continue
            self.entries.append(
                MetricEntry(step, wall_time_ms, names[i], values[i])
            )
        self._total_logged += n
        if len(self.entries) >= self.buffer_size:
            self.flush()

    def flush(mut self) raises:
        if self.server_url.byte_length() == 0 or len(self.entries) == 0:
            return

        if not self._run_registered:
            self._register_run()
            self._run_registered = True

        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), self.run_id)
        w.key(String("metrics"))
        w.begin_array()
        for i in range(len(self.entries)):
            var e = self.entries[i].copy()
            w.begin_object()
            w.member(String("step"), e.step)
            w.member(String("wall_time_ms"), e.wall_time_ms)
            w.member(String("name"), e.name)
            w.member(String("value"), e.value)
            w.end_object()
        w.end_array()
        w.end_object()

        self._post(self.server_url.removesuffix("/") + "/ingest", w.done())
        self.entries.clear()

    def close(mut self) raises:
        """Flush, then drain the sink and join its thread.

        ⚠ THE DRAIN IS BOUNDED. `drain_ms` is a budget, not a promise — see
        `HttpPostSink`. A hung dashboard is bounded by the worker's `dead`
        latch instead, at one client timeout rather than one per payload.
        """
        self.flush()
        if self._sink:
            self._report_transport_once()
            var line = self.sink_report()
            self._sink.value().close(drain_ms=3000)
            var final = self.sink_report()
            if final.byte_length() > 0:
                print(final)
            elif line.byte_length() > 0:
                print(line)

    def set_config(mut self, key: String, value: String):
        for i in range(len(self._config_keys)):
            if self._config_keys[i] == key:
                self._config_vals[i] = value
                return
        self._config_keys.append(key)
        self._config_vals.append(value)

    def is_active(self) -> Bool:
        return self.server_url.byte_length() > 0

    def _register_run(mut self) raises:
        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), self.run_id)
        w.member(String("run_name"), self.run_name)
        w.key(String("config"))
        w.begin_object()
        for i in range(len(self._config_keys)):
            w.member(self._config_keys[i], self._config_vals[i])
        w.end_object()
        w.end_object()
        self._post(self.server_url.removesuffix("/") + "/runs", w.done())

    def _post(mut self, url: String, payload: String):
        """Queue a JSON POST. Returns immediately; the network happens on the
        sink's thread.

        ⚠ EVERY FAILURE IS SWALLOWED. A dead or slow dashboard must not be able
        to kill a training run or flood its stdout, so this reports at most
        once and returns. That is a deliberate asymmetry with the rest of
        `io/`, where a failed transfer raises.

        Measured: twenty flushes against a dashboard answering in 100ms cost
        **2090 ms** of training time synchronously and **0.7 ms** through the
        sink, arriving byte-identical and in order
        (`docs/design_spikes/spike_async_post_spsc_ring.mojo`).
        """
        try:
            if not self._sink:
                self._sink = Optional(
                    HttpPostSink(api_key=self.api_key, timeout_ms=5000)
                )
            _ = self._sink.value().post(url, payload)
        except e:
            if not self._reported:
                self._reported = True
                print("  [logger] could not start the POST sink: " + String(e))
        self._report_transport_once()

    def _report_transport_once(mut self):
        """Print the first transport problem the worker recorded, once.

        The worker never prints: it runs on another thread and would interleave
        with training output. It records into atomic cells and the owning
        thread reports here.
        """
        if self._reported or not self._sink:
            return
        var s = self._sink.value()
        if s.shim_missing():
            self._reported = True
            print(
                "  [logger] the HTTP shim is missing — build it with"
                " `pixi run build-http`. Metrics will be dropped."
            )
        elif s.dead():
            self._reported = True
            print(
                "  [logger] the dashboard transport failed (last status "
                + String(s.last_status())
                + "); metrics will be dropped for the rest of this run."
            )

    def sink_report(self) -> String:
        """One line of delivery accounting, or empty if nothing was sent.

        ⚠ THE DROP COUNT IS THE COST OF THE DROP POLICY AND MUST BE VISIBLE. A
        Sink that never reports it is silently lossy; `close()` prints this.
        """
        if not self._sink:
            return String("")
        var s = self._sink.value()
        var total = s.sent() + s.failed() + s.dropped() + s.abandoned()
        if total == 0:
            return String("")
        return (
            "  [logger] "
            + String(s.sent())
            + " batches delivered, "
            + String(s.failed())
            + " failed, "
            + String(s.dropped())
            + " dropped (queue full), "
            + String(s.abandoned())
            + " abandoned at close"
        )

    def total_logged(self) -> Int:
        return self._total_logged

    def pending(self) -> Int:
        return len(self.entries)


# =============================================================================
# CompositeLogger — fan-out to two loggers
# =============================================================================


struct CompositeLogger[A: Logger, B: Logger](Logger):
    """Fans out log calls to two underlying loggers.

    Usage:
        var logger = CompositeLogger(
            CsvLogger("logs/run.csv"),
            RemoteLogger(server_url="http://host:3000/api"),
        )
    """

    var a: Self.A
    var b: Self.B

    def __init__(out self, a: Self.A, b: Self.B):
        self.a = a.copy()
        self.b = b.copy()

    def __init__(out self, *, deinit move: Self):
        self.a = move.a^
        self.b = move.b^

    def __init__(out self, *, copy: Self):
        self.a = copy.a.copy()
        self.b = copy.b.copy()

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        self.a.log_scalar(name, value, step)
        self.b.log_scalar(name, value, step)

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        self.a.log_scalars(names, values, step)
        self.b.log_scalars(names, values, step)

    def flush(mut self) raises:
        self.a.flush()
        self.b.flush()

    def close(mut self) raises:
        self.a.close()
        self.b.close()

    def set_config(mut self, key: String, value: String):
        self.a.set_config(key, value)
        self.b.set_config(key, value)

    def is_active(self) -> Bool:
        return True

    def __deinit__(deinit self):
        _ = self.a^
        _ = self.b^



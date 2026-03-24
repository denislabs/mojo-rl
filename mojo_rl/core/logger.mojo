"""Trait-based training metrics logger with pluggable backends.

Logger trait defines the interface. Concrete implementations:
  - NoOpLogger: does nothing (zero overhead, default)
  - CsvLogger: appends CSV rows to a local file
  - RemoteLogger: POSTs JSON batches to an HTTP server
  - CompositeLogger[A, B]: fans out to two loggers

Collection is pure Mojo with near-zero overhead. The remote backend
uses Python `urllib` via Mojo's Python interop (only during flush).

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
from std.python import Python, PythonObject
from std.math import isnan, isinf

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

    def __init__(out self, *, deinit take: Self):
        self.step = take.step
        self.wall_time_ms = take.wall_time_ms
        self.name = take.name^
        self.value = take.value


# =============================================================================
# Logger Trait
# =============================================================================


trait Logger(Copyable, ImplicitlyDestructible, Movable):
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

    def __init__(out self, *, deinit take: Self):
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
    var _start_ns: UInt
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

    def __init__(out self, *, deinit take: Self):
        self.file_path = take.file_path^
        self.entries = take.entries^
        self.buffer_size = take.buffer_size
        self._start_ns = take._start_ns
        self._file_header_written = take._file_header_written
        self._total_logged = take._total_logged

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
    the run at `server_url/runs` on first flush.  Uses Python urllib.
    """

    var run_id: String
    var run_name: String
    var server_url: String
    var api_key: String
    var entries: List[MetricEntry]
    var buffer_size: Int
    var _start_ns: UInt
    var _config_keys: List[String]
    var _config_vals: List[String]
    var _run_registered: Bool
    var _total_logged: Int

    def __init__(
        out self,
        server_url: String,
        run_name: String = "",
        run_id: String = "",
        buffer_size: Int = 200,
        api_key: String = "",
    ):
        self._start_ns = perf_counter_ns()
        if len(run_id) > 0:
            self.run_id = run_id
        else:
            self.run_id = "run_" + String(self._start_ns)
        self.run_name = run_name if len(run_name) > 0 else self.run_id
        self.server_url = server_url
        self.api_key = api_key
        self.entries = List[MetricEntry]()
        self.buffer_size = buffer_size
        self._config_keys = List[String]()
        self._config_vals = List[String]()
        self._run_registered = False
        self._total_logged = 0

    def __init__(out self, *, deinit take: Self):
        self.run_id = take.run_id^
        self.run_name = take.run_name^
        self.server_url = take.server_url^
        self.api_key = take.api_key^
        self.entries = take.entries^
        self.buffer_size = take.buffer_size
        self._start_ns = take._start_ns
        self._config_keys = take._config_keys^
        self._config_vals = take._config_vals^
        self._run_registered = take._run_registered
        self._total_logged = take._total_logged

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
        from std.python import Python

        var json_mod = Python.import_module("json")
        var urllib_request = Python.import_module("urllib.request")

        if not self._run_registered:
            self._register_run(json_mod, urllib_request)
            self._run_registered = True

        var metrics_list = Python.evaluate("[]")
        for i in range(len(self.entries)):
            var e = self.entries[i].copy()
            var entry = Python.evaluate("{}")
            entry["step"] = e.step
            entry["wall_time_ms"] = e.wall_time_ms
            entry["name"] = PythonObject(e.name)
            entry["value"] = e.value
            metrics_list.append(entry)

        var payload = Python.evaluate("{}")
        payload["run_id"] = PythonObject(self.run_id)
        payload["metrics"] = metrics_list

        var url = self.server_url.removesuffix("/") + "/ingest"
        _http_post(urllib_request, json_mod, url, payload, self.api_key)
        self.entries.clear()

    def close(mut self) raises:
        self.flush()

    def set_config(mut self, key: String, value: String):
        for i in range(len(self._config_keys)):
            if self._config_keys[i] == key:
                self._config_vals[i] = value
                return
        self._config_keys.append(key)
        self._config_vals.append(value)

    def is_active(self) -> Bool:
        return True

    def _register_run(
        mut self,
        json_mod: PythonObject,
        urllib_request: PythonObject,
    ) raises:
        var config = Python.evaluate("{}")
        for i in range(len(self._config_keys)):
            config[PythonObject(self._config_keys[i])] = PythonObject(
                self._config_vals[i]
            )
        var payload = Python.evaluate("{}")
        payload["run_id"] = PythonObject(self.run_id)
        payload["run_name"] = PythonObject(self.run_name)
        payload["config"] = config
        var url = self.server_url.removesuffix("/") + "/runs"
        _http_post(urllib_request, json_mod, url, payload, self.api_key)

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

    def __init__(out self, *, deinit take: Self):
        self.a = take.a^
        self.b = take.b^

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

    def __del__(deinit self):
        _ = self.a^
        _ = self.b^


# =============================================================================
# HTTP Helper
# =============================================================================


def _http_post(
    urllib_request: PythonObject,
    json_mod: PythonObject,
    url: String,
    payload: PythonObject,
    api_key: String = "",
) raises:
    """POST JSON payload to a URL. Silently ignores errors to avoid
    disrupting training if the server is down."""
    try:
        var data = json_mod.dumps(payload).encode("utf-8")
        var req = urllib_request.Request(
            PythonObject(url),
            data=data,
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "mojo-rl/1.0")
        if len(api_key) > 0:
            req.add_header("Authorization", "Bearer " + api_key)
        _ = urllib_request.urlopen(req, timeout=5)
    except e:
        print("  [logger] POST", url, "failed:", String(e))

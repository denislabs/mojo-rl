"""Training metrics logger with file and remote backends.

Collects named scalar time series during training and flushes them
periodically to a local CSV file and/or a remote HTTP server.

Collection is pure Mojo with near-zero overhead. The remote backend
uses Python `urllib` via Mojo's Python interop (only during flush).

Usage:
    var logger = MetricsLogger(
        file_path="logs/run_001.csv",     # local CSV (empty = disabled)
        server_url="http://host:3000/api", # remote POST (empty = disabled)
        run_name="ppo_halfcheetah_v3",
        buffer_size=200,                   # flush every 200 entries
    )
    logger.set_config("algorithm", "PPO")
    logger.set_config("environment", "HalfCheetah")
    logger.set_config("lr", "3e-4")

    # In training loop:
    logger.log_scalar("reward", avg_reward, step)
    logger.log_scalar("loss", loss_val, step)

    # At end:
    logger.close()  # final flush
"""

from std.time import perf_counter_ns
from std.python import Python, PythonObject


# =============================================================================
# MetricEntry — single buffered data point
# =============================================================================


struct MetricEntry(Copyable, Movable):
    """A single scalar metric data point."""

    var step: Int
    var wall_time_ms: Float64
    var name: String
    var value: Float64

    fn __init__(
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

    fn __init__(out self, *, copy: Self):
        self.step = copy.step
        self.wall_time_ms = copy.wall_time_ms
        self.name = copy.name
        self.value = copy.value

    fn __init__(out self, *, deinit take: Self):
        self.step = take.step
        self.wall_time_ms = take.wall_time_ms
        self.name = take.name^
        self.value = take.value


# =============================================================================
# MetricsLogger
# =============================================================================


struct MetricsLogger(Movable):
    """Buffered metrics logger with file and remote backends.

    Accumulates MetricEntry objects in memory and flushes them when the
    buffer reaches `buffer_size` entries, or when `flush()` / `close()`
    is called explicitly.

    Backends (both optional, set via constructor args):
      - **File**: Appends CSV rows to `file_path`. Format:
            step,wall_time_ms,name,value
      - **Remote**: POSTs JSON batches to `server_url/ingest`.
            Run registration POSTed to `server_url/runs` on first flush.

    The remote backend requires Python (urllib) but is only invoked
    during flush — the hot logging path is pure Mojo.
    """

    var run_id: String
    var run_name: String

    # Backends
    var file_path: String
    var server_url: String

    # Buffer
    var entries: List[MetricEntry]
    var buffer_size: Int

    # Timing
    var _start_ns: UInt

    # Run config (algorithm, env, hyperparams, etc.)
    var _config_keys: List[String]
    var _config_vals: List[String]

    # State
    var _run_registered: Bool
    var _file_header_written: Bool
    var _total_logged: Int

    fn __init__(
        out self,
        run_name: String = "",
        file_path: String = "",
        server_url: String = "",
        run_id: String = "",
        buffer_size: Int = 200,
    ):
        """Initialize the metrics logger.

        Args:
            run_name: Human-readable name for this run.
            file_path: Path to CSV log file (empty to disable file logging).
            server_url: Base URL of the dashboard server (empty to disable).
            run_id: Unique run identifier. Auto-generated from timestamp if empty.
            buffer_size: Number of entries to buffer before auto-flushing.
        """
        self._start_ns = perf_counter_ns()

        if len(run_id) > 0:
            self.run_id = run_id
        else:
            # Generate run_id from timestamp (nanosecond counter as hex)
            self.run_id = "run_" + String(self._start_ns)

        self.run_name = run_name if len(run_name) > 0 else self.run_id
        self.file_path = file_path
        self.server_url = server_url
        self.entries = List[MetricEntry]()
        self.buffer_size = buffer_size
        self._config_keys = List[String]()
        self._config_vals = List[String]()
        self._run_registered = False
        self._file_header_written = False
        self._total_logged = 0

    fn __init__(out self, *, deinit take: Self):
        self.run_id = take.run_id^
        self.run_name = take.run_name^
        self.file_path = take.file_path^
        self.server_url = take.server_url^
        self.entries = take.entries^
        self.buffer_size = take.buffer_size
        self._start_ns = take._start_ns
        self._config_keys = take._config_keys^
        self._config_vals = take._config_vals^
        self._run_registered = take._run_registered
        self._file_header_written = take._file_header_written
        self._total_logged = take._total_logged

    # =========================================================================
    # Configuration
    # =========================================================================

    fn set_config(mut self, key: String, value: String):
        """Attach a config key-value pair to this run.

        Config is sent to the remote server on run registration.
        Common keys: "algorithm", "environment", "lr", "gamma", "batch_size".

        Args:
            key: Config parameter name.
            value: Config parameter value (as string).
        """
        # Update existing key if present
        for i in range(len(self._config_keys)):
            if self._config_keys[i] == key:
                self._config_vals[i] = value
                return
        self._config_keys.append(key)
        self._config_vals.append(value)

    # =========================================================================
    # Logging
    # =========================================================================

    fn log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        """Log a single scalar metric.

        This is the primary logging method. Called from training loops
        to record losses, rewards, Q-values, epsilon, etc.

        Auto-flushes when buffer reaches `buffer_size` entries.

        Args:
            name: Metric name (e.g. "reward", "loss", "q_mean").
            value: Scalar value.
            step: Training step (env transitions or gradient steps).
        """
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0

        self.entries.append(MetricEntry(step, wall_time_ms, name, value))
        self._total_logged += 1

        if len(self.entries) >= self.buffer_size:
            self.flush()

    fn log_scalars(
        mut self,
        names: List[String],
        values: List[Float64],
        step: Int,
    ) raises:
        """Log multiple scalar metrics at the same step.

        Convenience method for logging several metrics with the same
        step and wall_time. More efficient than multiple log_scalar calls
        (single timestamp, single buffer-size check).

        Args:
            names: Metric names.
            values: Corresponding values (must be same length as names).
            step: Training step.
        """
        var elapsed_ns = perf_counter_ns() - self._start_ns
        var wall_time_ms = Float64(elapsed_ns) / 1_000_000.0

        var n = min(len(names), len(values))
        for i in range(n):
            self.entries.append(
                MetricEntry(step, wall_time_ms, names[i], values[i])
            )
        self._total_logged += n

        if len(self.entries) >= self.buffer_size:
            self.flush()

    # =========================================================================
    # Flush
    # =========================================================================

    fn flush(mut self) raises:
        """Flush buffered entries to all enabled backends.

        Called automatically when buffer is full, or explicitly by the user.
        After flush, the buffer is cleared.
        """
        if len(self.entries) == 0:
            return

        if len(self.file_path) > 0:
            self._flush_file()

        if len(self.server_url) > 0:
            self._flush_remote()

        self.entries.clear()

    fn close(mut self) raises:
        """Final flush and cleanup. Call at the end of training."""
        self.flush()

    # =========================================================================
    # File Backend
    # =========================================================================

    fn _flush_file(mut self) raises:
        """Append buffered entries to CSV file."""
        var content = String("")

        # Write header on first flush
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

        # Append to file
        with open(self.file_path, "a") as f:
            f.write(content)

    # =========================================================================
    # Remote Backend
    # =========================================================================

    fn _flush_remote(mut self) raises:
        """POST buffered entries to the remote dashboard server."""
        from std.python import Python

        var json_mod = Python.import_module("json")
        var urllib_request = Python.import_module("urllib.request")

        # Register run on first flush
        if not self._run_registered:
            self._register_run(json_mod, urllib_request)
            self._run_registered = True

        # Build metrics payload
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

        var url = self.server_url + "/ingest"
        _http_post(urllib_request, json_mod, url, payload)

    fn _register_run(
        mut self,
        json_mod: PythonObject,
        urllib_request: PythonObject,
    ) raises:
        """POST run registration with config metadata."""
        var config = Python.evaluate("{}")
        for i in range(len(self._config_keys)):
            config[PythonObject(self._config_keys[i])] = PythonObject(
                self._config_vals[i]
            )

        var payload = Python.evaluate("{}")
        payload["run_id"] = PythonObject(self.run_id)
        payload["run_name"] = PythonObject(self.run_name)
        payload["config"] = config

        var url = self.server_url + "/runs"
        _http_post(urllib_request, json_mod, url, payload)

    # =========================================================================
    # Stats
    # =========================================================================

    fn total_logged(self) -> Int:
        """Return total number of data points logged (including flushed)."""
        return self._total_logged

    fn pending(self) -> Int:
        """Return number of entries waiting in buffer."""
        return len(self.entries)


# =============================================================================
# LoggerPtr — nullable pointer for optional logger in training loops
# =============================================================================


comptime LoggerPtr = UnsafePointer[MetricsLogger, MutAnyOrigin]
"""Nullable pointer to a MetricsLogger.

Used as a parameter in training loop functions to make logging optional.
Default value is null (no logging). Callers who want logging pass
`UnsafePointer(to=logger)`.

Usage in training loops:
    fn run_train(..., logger: LoggerPtr = LoggerPtr()) raises:
        _log(logger, "reward", avg_reward, step)

Usage by callers:
    var logger = MetricsLogger(file_path="logs/run.csv")
    run_train(..., logger=UnsafePointer(to=logger))
"""


fn _log(
    logger: LoggerPtr, name: String, value: Float64, step: Int
) raises:
    """Log a scalar if logger is not null. No-op otherwise."""
    if logger:
        logger[].log_scalar(name, value, step)


fn _log_flush(logger: LoggerPtr) raises:
    """Flush logger if not null. No-op otherwise."""
    if logger:
        logger[].flush()


# =============================================================================
# HTTP Helper
# =============================================================================


fn _http_post(
    urllib_request: PythonObject,
    json_mod: PythonObject,
    url: String,
    payload: PythonObject,
) raises:
    """POST JSON payload to a URL. Silently ignores errors to avoid
    disrupting training if the server is down.

    Args:
        urllib_request: Python urllib.request module.
        json_mod: Python json module.
        url: Target URL.
        payload: Python dict to serialize as JSON.
    """
    try:
        var data = json_mod.dumps(payload).encode("utf-8")
        var req = urllib_request.Request(
            PythonObject(url),
            data=data,
        )
        req.add_header("Content-Type", "application/json")
        _ = urllib_request.urlopen(req, timeout=5)
    except:
        # Silently ignore network errors to not disrupt training
        pass

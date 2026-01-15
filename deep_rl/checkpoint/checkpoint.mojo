"""Checkpoint utilities for saving and loading model state.

This module provides utilities for saving/loading neural network parameters,
optimizer state, and agent state to/from text files.

File format:
    # mojo-rl checkpoint v1
    # type: network|trainer|agent
    # param_size: N
    # state_size: M
    # dtype: float32
    params:
    0.123456
    -0.789012
    ...
    optimizer_state:
    0.0
    0.0
    ...
    metadata:
    key1=value1
    key2=value2

Usage:
    # Saving
    save_checkpoint("model.ckpt", params, optimizer_state, metadata)

    # Loading
    var header = read_checkpoint_header("model.ckpt")
    var params = read_float_section("model.ckpt", "params:", header.param_size)
    var state = read_float_section("model.ckpt", "optimizer_state:", header.state_size)
"""

from ..constants import dtype


struct CheckpointHeader(Copyable, Movable):
    """Header information from a checkpoint file."""

    var version: Int
    var checkpoint_type: String
    var param_size: Int
    var state_size: Int
    var dtype_str: String

    fn __init__(out self):
        self.version = 1
        self.checkpoint_type = "network"
        self.param_size = 0
        self.state_size = 0
        self.dtype_str = "float32"

    fn __init__(
        out self,
        version: Int,
        checkpoint_type: String,
        param_size: Int,
        state_size: Int,
        dtype_str: String,
    ):
        self.version = version
        self.checkpoint_type = checkpoint_type
        self.param_size = param_size
        self.state_size = state_size
        self.dtype_str = dtype_str

    fn __copyinit__(out self, existing: Self):
        self.version = existing.version
        self.checkpoint_type = existing.checkpoint_type
        self.param_size = existing.param_size
        self.state_size = existing.state_size
        self.dtype_str = existing.dtype_str

    fn __moveinit__(out self, deinit existing: Self):
        self.version = existing.version
        self.checkpoint_type = existing.checkpoint_type^
        self.param_size = existing.param_size
        self.state_size = existing.state_size
        self.dtype_str = existing.dtype_str^


fn write_checkpoint_header(
    checkpoint_type: String,
    param_size: Int,
    state_size: Int,
) -> String:
    """Generate checkpoint header string.

    Args:
        checkpoint_type: Type of checkpoint ("network", "trainer", or "agent").
        param_size: Number of parameters.
        state_size: Size of optimizer state.

    Returns:
        Header string to write to file.
    """
    var header = String("# mojo-rl checkpoint v1\n")
    header += "# type: " + checkpoint_type + "\n"
    header += "# param_size: " + String(param_size) + "\n"
    header += "# state_size: " + String(state_size) + "\n"
    header += "# dtype: float32\n"
    return header


fn write_float_section[
    SIZE: Int
](section_name: String, data: InlineArray[Scalar[dtype], SIZE]) -> String:
    """Generate string for a float array section.

    Args:
        section_name: Name of the section (e.g., "params:" or "optimizer_state:").
        data: Array of float values.

    Returns:
        Section string to write to file.
    """
    var content = section_name + "\n"
    for i in range(SIZE):
        content += String(Float64(data[i])) + "\n"
    return content


fn write_float_section_list(
    section_name: String, data: List[Scalar[dtype]]
) -> String:
    """Generate string for a float array section from List.

    Args:
        section_name: Name of the section (e.g., "params:" or "optimizer_state:").
        data: List of float values.

    Returns:
        Section string to write to file.
    """
    var content = section_name + "\n"
    for i in range(len(data)):
        content += String(Float64(data[i])) + "\n"
    return content


fn write_metadata_section(metadata: List[String]) -> String:
    """Generate string for metadata section.

    Args:
        metadata: List of key=value strings.

    Returns:
        Metadata section string.
    """
    var content = String("metadata:\n")
    for i in range(len(metadata)):
        content += metadata[i] + "\n"
    return content


fn read_checkpoint_file(filepath: String) raises -> String:
    """Read entire checkpoint file contents.

    Args:
        filepath: Path to the checkpoint file.

    Returns:
        File contents as string.
    """
    with open(filepath, "r") as f:
        return f.read()


fn split_lines(content: String) -> List[String]:
    """Split content into lines.

    Args:
        content: String content to split.

    Returns:
        List of lines.
    """
    var lines = List[String]()
    var current_line = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == ord("\n"):
            lines.append(current_line)
            current_line = String("")
        else:
            current_line += chr(Int(c))
    if len(current_line) > 0:
        lines.append(current_line)
    return lines^


fn parse_checkpoint_header(content: String) raises -> CheckpointHeader:
    """Parse checkpoint header from file content.

    Args:
        content: File content string.

    Returns:
        CheckpointHeader with parsed values.

    Raises:
        If file format is invalid.
    """
    var lines = split_lines(content)
    var header = CheckpointHeader()

    for i in range(len(lines)):
        var line = lines[i]
        if line.startswith("# mojo-rl checkpoint v"):
            # Parse version
            var version_str = String(line[len("# mojo-rl checkpoint v") :])
            header.version = Int(atol(version_str))
        elif line.startswith("# type: "):
            header.checkpoint_type = String(line[len("# type: ") :])
        elif line.startswith("# param_size: "):
            var size_str = String(line[len("# param_size: ") :])
            header.param_size = Int(atol(size_str))
        elif line.startswith("# state_size: "):
            var size_str = String(line[len("# state_size: ") :])
            header.state_size = Int(atol(size_str))
        elif line.startswith("# dtype: "):
            header.dtype_str = String(line[len("# dtype: ") :])
        elif line == "params:":
            # End of header
            break

    return header^


fn find_section_start(lines: List[String], section_name: String) -> Int:
    """Find the line index where a section starts.

    Args:
        lines: List of file lines.
        section_name: Section name to find (e.g., "params:").

    Returns:
        Line index after section header, or -1 if not found.
    """
    for i in range(len(lines)):
        if lines[i] == section_name:
            return i + 1
    return -1


fn read_float_section[
    SIZE: Int
](content: String, section_name: String) raises -> InlineArray[Scalar[dtype], SIZE]:
    """Read a float array section from checkpoint content.

    Args:
        content: File content string.
        section_name: Section to read (e.g., "params:" or "optimizer_state:").

    Returns:
        InlineArray with the parsed values.
    """
    var lines = split_lines(content)
    var start_idx = find_section_start(lines, section_name)

    var result = InlineArray[Scalar[dtype], SIZE](uninitialized=True)

    if start_idx < 0:
        # Section not found, return zeros
        for i in range(SIZE):
            result[i] = 0
        return result

    for i in range(SIZE):
        var line_idx = start_idx + i
        if line_idx >= len(lines):
            result[i] = 0
        else:
            var line = lines[line_idx]
            if len(line) == 0 or line.startswith("#") or line.endswith(":"):
                result[i] = 0
            else:
                result[i] = Scalar[dtype](atof(line))

    return result


fn read_float_section_list(
    content: String, section_name: String, size: Int
) raises -> List[Scalar[dtype]]:
    """Read a float array section from checkpoint content into a List.

    Args:
        content: File content string.
        section_name: Section to read (e.g., "params:" or "optimizer_state:").
        size: Number of values to read.

    Returns:
        List with the parsed values.
    """
    var lines = split_lines(content)
    var start_idx = find_section_start(lines, section_name)

    var result = List[Scalar[dtype]](capacity=size)

    if start_idx < 0:
        # Section not found, return zeros
        for i in range(size):
            result.append(0)
        return result^

    for i in range(size):
        var line_idx = start_idx + i
        if line_idx >= len(lines):
            result.append(0)
        else:
            var line = lines[line_idx]
            if len(line) == 0 or line.startswith("#") or line.endswith(":"):
                result.append(0)
            else:
                result.append(Scalar[dtype](atof(line)))

    return result^


fn read_metadata_section(content: String) raises -> List[String]:
    """Read metadata section from checkpoint content.

    Args:
        content: File content string.

    Returns:
        List of key=value strings.
    """
    var lines = split_lines(content)
    var start_idx = find_section_start(lines, "metadata:")

    var result = List[String]()

    if start_idx < 0:
        return result^

    for i in range(start_idx, len(lines)):
        var line = lines[i]
        if len(line) == 0:
            continue
        if line.startswith("#") or line.endswith(":"):
            break
        if line.find("=") >= 0:
            result.append(line)

    return result^


fn get_metadata_value(metadata: List[String], key: String) -> String:
    """Get value for a key from metadata list.

    Args:
        metadata: List of key=value strings.
        key: Key to look for.

    Returns:
        Value string, or empty string if not found.
    """
    var prefix = key + "="
    for i in range(len(metadata)):
        if metadata[i].startswith(prefix):
            return String(metadata[i][len(prefix) :])
    return String("")


fn save_checkpoint_file(filepath: String, content: String) raises:
    """Write checkpoint content to file.

    Args:
        filepath: Path to write to.
        content: Content to write.
    """
    with open(filepath, "w") as f:
        f.write(content)

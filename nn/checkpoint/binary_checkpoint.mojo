"""Binary checkpoint format for compact model serialization.

Stores float data as raw float32 bytes instead of text, achieving ~3x smaller
files compared to the text format.

Binary format layout:
    4 bytes:  Magic "MRCK" (Mojo RL ChecKpoint)
    4 bytes:  Version (uint32 LE) = 2
    4 bytes:  checkpoint_type name length (uint32 LE)
    N bytes:  checkpoint_type name (UTF-8)
    4 bytes:  num_float_sections (uint32 LE)
    For each float section:
        4 bytes:  section name length (uint32 LE)
        N bytes:  section name (UTF-8)
        4 bytes:  count (uint32 LE, number of float32 values)
        count*4:  float32 data (raw bytes, native endian)
    4 bytes:  num_metadata_entries (uint32 LE)
    For each metadata entry:
        4 bytes:  entry length (uint32 LE)
        N bytes:  entry (UTF-8 "key=value")

Usage:
    # Saving
    var ckpt = BinaryCheckpoint("dqn_agent")
    ckpt.add_float_section("online_params", params_list)
    ckpt.add_float_section("online_optimizer_state", state_list)
    ckpt.add_metadata("gamma", "0.99")
    ckpt.save("model.ckpt.bin")

    # Loading
    var ckpt = BinaryCheckpoint.load("model.ckpt.bin")
    var params = ckpt.get_float_section("online_params")
    var gamma = ckpt.get_metadata("gamma")
"""

from ..constants import dtype


struct FloatSection(Copyable, Movable):
    """A named section of float32 data."""

    var name: String
    var data: List[Scalar[dtype]]

    fn __init__(out self, name: String, data: List[Scalar[dtype]]):
        self.name = name
        self.data = data.copy()

    fn __init__(out self, *, copy: Self):
        self.name = copy.name
        self.data = copy.data.copy()

    fn __init__(out self, *, deinit take: Self):
        self.name = take.name^
        self.data = take.data^


# Magic bytes "MRCK"
comptime MAGIC_0: UInt8 = UInt8(ord("M"))
comptime MAGIC_1: UInt8 = UInt8(ord("R"))
comptime MAGIC_2: UInt8 = UInt8(ord("C"))
comptime MAGIC_3: UInt8 = UInt8(ord("K"))
comptime BINARY_VERSION: UInt32 = 2


struct BinaryCheckpoint(Copyable, Movable):
    """Binary checkpoint container for efficient model serialization.

    Holds named float sections and key=value metadata. Serializes float data
    as raw bytes for ~3x smaller files than the text format.
    """

    var checkpoint_type: String
    var sections: List[FloatSection]
    var metadata: List[String]

    fn __init__(out self, checkpoint_type: String = "network"):
        self.checkpoint_type = checkpoint_type
        self.sections = List[FloatSection]()
        self.metadata = List[String]()

    fn __init__(out self, *, copy: Self):
        self.checkpoint_type = copy.checkpoint_type
        self.sections = copy.sections.copy()
        self.metadata = copy.metadata.copy()

    fn __init__(out self, *, deinit take: Self):
        self.checkpoint_type = take.checkpoint_type^
        self.sections = take.sections^
        self.metadata = take.metadata^

    # =========================================================================
    # Building
    # =========================================================================

    fn add_float_section(mut self, name: String, data: List[Scalar[dtype]]):
        """Add a named float section.

        Args:
            name: Section name (e.g., "actor_params", "critic_optimizer_state").
            data: Float values to store.
        """
        self.sections.append(FloatSection(name, data))

    fn add_metadata(mut self, key: String, value: String):
        """Add a metadata key=value pair.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata.append(key + "=" + value)

    fn add_metadata_entry(mut self, entry: String):
        """Add a raw metadata entry (already formatted as key=value).

        Args:
            entry: Pre-formatted "key=value" string.
        """
        self.metadata.append(entry)

    # =========================================================================
    # Querying
    # =========================================================================

    fn get_float_section(
        self, name: String, size: Int
    ) raises -> List[Scalar[dtype]]:
        """Get float data for a named section.

        Args:
            name: Section name to look up.
            size: Expected number of values (returns zeros if section not found
                  or has fewer values).

        Returns:
            List of float values, zero-padded to `size` if needed.
        """
        for i in range(len(self.sections)):
            if self.sections[i].name == name:
                ref section = self.sections[i]
                if len(section.data) >= size:
                    # Return first `size` values
                    var result = List[Scalar[dtype]](capacity=size)
                    for j in range(size):
                        result.append(section.data[j])
                    return result^
                else:
                    # Pad with zeros
                    var result = List[Scalar[dtype]](capacity=size)
                    for j in range(len(section.data)):
                        result.append(section.data[j])
                    for _ in range(size - len(section.data)):
                        result.append(0)
                    return result^

        # Section not found — return zeros
        var result = List[Scalar[dtype]](capacity=size)
        for _ in range(size):
            result.append(0)
        return result^

    fn get_metadata_value(self, key: String) -> String:
        """Get value for a metadata key.

        Args:
            key: Key to look for.

        Returns:
            Value string, or empty string if not found.
        """
        var prefix = key + "="
        for i in range(len(self.metadata)):
            if self.metadata[i].startswith(prefix):
                return String(self.metadata[i][len(prefix) :])
        return String("")

    fn get_metadata_list(self) -> List[String]:
        """Get all metadata entries.

        Returns:
            List of "key=value" strings.
        """
        return self.metadata.copy()

    # =========================================================================
    # Serialization
    # =========================================================================

    fn to_bytes(self) -> List[UInt8]:
        """Serialize to binary format.

        Returns:
            Complete binary checkpoint as byte list.
        """
        var buf = List[UInt8]()

        # Magic
        buf.append(MAGIC_0)
        buf.append(MAGIC_1)
        buf.append(MAGIC_2)
        buf.append(MAGIC_3)

        # Version
        _write_uint32(buf, BINARY_VERSION)

        # Checkpoint type
        var type_bytes = self.checkpoint_type.as_bytes()
        _write_uint32(buf, UInt32(len(type_bytes)))
        for i in range(len(type_bytes)):
            buf.append(type_bytes[i])

        # Float sections
        _write_uint32(buf, UInt32(len(self.sections)))
        for i in range(len(self.sections)):
            ref section = self.sections[i]
            # Section name
            var name_bytes = section.name.as_bytes()
            _write_uint32(buf, UInt32(len(name_bytes)))
            for j in range(len(name_bytes)):
                buf.append(name_bytes[j])
            # Float count
            var count = len(section.data)
            _write_uint32(buf, UInt32(count))
            # Float data as raw bytes
            var fptr = section.data.unsafe_ptr().bitcast[UInt8]()
            for j in range(count * 4):
                buf.append((fptr + j)[])

        # Metadata entries
        _write_uint32(buf, UInt32(len(self.metadata)))
        for i in range(len(self.metadata)):
            var entry_bytes = self.metadata[i].as_bytes()
            _write_uint32(buf, UInt32(len(entry_bytes)))
            for j in range(len(entry_bytes)):
                buf.append(entry_bytes[j])

        return buf^

    @staticmethod
    fn from_bytes(data: List[UInt8]) raises -> BinaryCheckpoint:
        """Deserialize from binary format.

        Args:
            data: Binary checkpoint data.

        Returns:
            Parsed BinaryCheckpoint.

        Raises:
            If data is too short or magic bytes don't match.
        """
        var pos: Int = 0

        # Verify magic
        if len(data) < 4:
            raise Error("Binary checkpoint too short")
        if (
            data[0] != MAGIC_0
            or data[1] != MAGIC_1
            or data[2] != MAGIC_2
            or data[3] != MAGIC_3
        ):
            raise Error("Invalid binary checkpoint magic bytes")
        pos = 4

        # Version
        var version = _read_uint32(data, pos)
        pos += 4
        if version != BINARY_VERSION:
            raise Error(
                "Unsupported binary checkpoint version: " + String(version)
            )

        # Checkpoint type
        var type_len = Int(_read_uint32(data, pos))
        pos += 4
        var checkpoint_type = _read_string(data, pos, type_len)
        pos += type_len

        var ckpt = BinaryCheckpoint(checkpoint_type)

        # Float sections
        var num_sections = Int(_read_uint32(data, pos))
        pos += 4
        for _ in range(num_sections):
            # Section name
            var name_len = Int(_read_uint32(data, pos))
            pos += 4
            var name = _read_string(data, pos, name_len)
            pos += name_len

            # Float count
            var count = Int(_read_uint32(data, pos))
            pos += 4

            # Float data
            var section_data = List[Scalar[dtype]](capacity=count)
            var fptr = (data.unsafe_ptr() + pos).bitcast[Scalar[dtype]]()
            for j in range(count):
                section_data.append((fptr + j)[])
            pos += count * 4

            ckpt.sections.append(FloatSection(name, section_data^))

        # Metadata entries
        if pos + 4 <= len(data):
            var num_metadata = Int(_read_uint32(data, pos))
            pos += 4
            for _ in range(num_metadata):
                var entry_len = Int(_read_uint32(data, pos))
                pos += 4
                var entry = _read_string(data, pos, entry_len)
                pos += entry_len
                ckpt.metadata.append(entry)

        return ckpt^

    fn estimated_text_size(self) -> Int:
        """Estimate how large the equivalent text format would be.

        Useful for reporting compression ratio.

        Returns:
            Estimated byte count of text format.
        """
        var size = 100  # header estimate
        for i in range(len(self.sections)):
            ref section = self.sections[i]
            # ~12 bytes per float in text (e.g. "-0.123456\n")
            size += len(section.name) + 2 + len(section.data) * 12
        for i in range(len(self.metadata)):
            size += len(self.metadata[i]) + 1
        return size

    # =========================================================================
    # File I/O
    # =========================================================================

    fn save(self, filepath: String) raises:
        """Save checkpoint to binary file.

        Args:
            filepath: Path to write to.
        """
        var data = self.to_bytes()
        with open(filepath, "w") as f:
            f.write_bytes(data)

    @staticmethod
    fn load(filepath: String) raises -> BinaryCheckpoint:
        """Load checkpoint from binary file.

        Args:
            filepath: Path to read from.

        Returns:
            Parsed BinaryCheckpoint.
        """
        with open(filepath, "r") as f:
            var data = f.read_bytes()
            return BinaryCheckpoint.from_bytes(data)


# =============================================================================
# Internal helpers
# =============================================================================


fn _write_uint32(mut buf: List[UInt8], value: UInt32):
    """Write a uint32 in little-endian to buffer."""
    buf.append(UInt8(value & 0xFF))
    buf.append(UInt8((value >> 8) & 0xFF))
    buf.append(UInt8((value >> 16) & 0xFF))
    buf.append(UInt8((value >> 24) & 0xFF))


fn _read_uint32(data: List[UInt8], pos: Int) -> UInt32:
    """Read a uint32 in little-endian from buffer."""
    return (
        UInt32(data[pos])
        | (UInt32(data[pos + 1]) << 8)
        | (UInt32(data[pos + 2]) << 16)
        | (UInt32(data[pos + 3]) << 24)
    )


fn _read_string(data: List[UInt8], pos: Int, length: Int) -> String:
    """Read a UTF-8 string from buffer."""
    var result = String("")
    for i in range(length):
        result += chr(Int(data[pos + i]))
    return result

from std.io.file import open


def read_file(path: String) -> String:
    try:
        return open(path, "r").read()
    except Exception:
        print("Error opening file")
        return ""


def test_comptime_parser_io() raises:
    # Parse XML at comptime
    comptime xml = read_file("test.xml")
    print(xml)


def main() raises:
    test_comptime_parser_io()

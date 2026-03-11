from std.io.file import open


fn read_file(path: String) -> String:
    try:
        return open(path, "r").read()
    except Exception:
        print("Error opening file")
        return ""


fn test_comptime_parser_io() raises:
    # Parse XML at comptime
    comptime xml = read_file("test.xml")
    print(xml)


fn main() raises:
    test_comptime_parser_io()

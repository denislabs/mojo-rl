from io.file import open
from testing import assert_true, TestSuite


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
    TestSuite.discover_tests[__functions_in_module()]().run()

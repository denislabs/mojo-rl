from io.file import open


fn read_file(path: String) -> String:
    try:
        return open(path, "r").read()
    except Exception:
        print("Error opening file")
        return ""


fn main():
    # Parse XML at comptime
    comptime xml = read_file("envs/half_cheetah/half_cheetah.xml")
    print(xml)

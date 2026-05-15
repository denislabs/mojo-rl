"""Smoke test: just confirm the libhdf5 FFI module imports and the
declared constants/types exist. Does not open any file."""

from mojo_rl.io.hdf5 import (
    H5File,
    H5Dataset,
    hid_t,
    hsize_t,
    herr_t,
    H5T_INTEGER,
    H5T_FLOAT,
    H5T_SGN_NONE,
    H5T_SGN_2,
    H5P_DEFAULT,
    H5S_ALL,
)


def main() raises:
    print("[smoke] hdf5 FFI module loaded.")
    print("        H5T_INTEGER =", H5T_INTEGER)
    print("        H5T_FLOAT   =", H5T_FLOAT)
    print("        H5T_SGN_2   =", H5T_SGN_2)
    print("        H5P_DEFAULT =", H5P_DEFAULT)
    print("        H5S_ALL     =", H5S_ALL)

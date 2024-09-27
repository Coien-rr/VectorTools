import numpy as np
import sys
import os


def print_meta_info(npts, dims, first):
    print(f"npts: {npts}")
    print(f"dims: {dims}")
    print(first)


def check_bin_file(path, data_type):
    (npts, dims) = np.fromfile(path, dtype=np.uint32, count=2, offset=0)
    first = np.fromfile(path, dtype=data_type, count=dims, offset=8)
    print_meta_info(npts, dims, first)


def check_fvecs_file(path, data_type):
    dims = np.fromfile(path, dtype=np.uint32, count=1, offset=0)[0]
    first = np.fromfile(path, dtype=data_type, count=dims, offset=4)
    npts = os.path.getsize(path) / (dims * 4 + 4)
    print_meta_info(npts, dims, first)


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 ./checkMetaInfo.py <dataset path> <format> <data type>")
        sys.exit(-1)

    dataset_path = sys.argv[1]
    file_format = sys.argv[2]
    data_type = sys.argv[3]

    if data_type == "float":
        read_type = np.float32
    elif data_type == "uint":
        read_type = np.uint32
    else:
        print("Unsupported data_type. Please use 'float' or 'uint'.")
        sys.exit(-1)

    if file_format == "bin":
        check_bin_file(dataset_path, read_type)
    elif file_format == "vecs":
        check_fvecs_file(dataset_path, read_type)
    else:
        print("Unsupported format. Please use 'bin' or 'vecs'.")
        sys.exit(-1)


if __name__ == "__main__":
    main()

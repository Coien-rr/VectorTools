import numpy as np
import sys
import os


def loadDataMetaInfo(sourceFile):
    (npts, dims) = np.fromfile(sourceFile, dtype=np.uint32, count=2, offset=0)
    print(f"Dataset MetaInfo: npts: {npts}, dims: {dims}")
    return (npts, dims)


def checkFileSize(sourceFile, npts, dims):
    actualFileSize = os.path.getsize(sourceFile)
    expectFileSize = npts * dims * 4 + 8
    if not actualFileSize == expectFileSize:
        print("Error: File size does not meet expectations, Please Follow Bin Format!")
        sys.exit(-1)


def loadMainData(sourceFile):
    (npts, dims) = np.fromfile(sourceFile, dtype=np.uint32, count=2, offset=0)
    data = np.fromfile(sourceFile, np.float32, count=-1, offset=8)
    data.reshape(npts, dims)
    print(data.shape)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 ./reorderData.py <source dataset path> <destination dataset path>"
        )
        sys.exit(-1)

    source = sys.argv[1]
    destination = sys.argv[2]

    (npts, dims) = loadDataMetaInfo(source)
    checkFileSize(source, npts, dims)
    loadMainData(source)


if __name__ == "__main__":
    main()

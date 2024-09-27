#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <string>
#include <sys/types.h>
#include <utility>

void checkFileExists(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("File not found: " + path);
  }
}

bool checkFileSize(const std::string &path, unsigned expectLen) {
  return std::filesystem::file_size(path) == expectLen;
}

std::fstream openBinaryFile(const std::string &path,
                            std::ios_base::openmode mode) {
  std::fstream reader(path, std::ios::binary | mode);
  if (!reader.is_open()) {
    std::cout << std::format("ERROR: Failed Open File [{}]", path) << std::endl;
    throw std::runtime_error("Failed to open file");
  }
  return std::move(reader);
}

template <typename T>
void loadMetaInfo(const std::string &path, unsigned &npts, unsigned &dims,
                  bool isBin) {
  std::fstream reader{openBinaryFile(path, std::ios::in)};
  if (isBin) {
    reader.read(reinterpret_cast<char *>(&npts), sizeof(unsigned));
  }
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));
  if (!isBin) {
    npts = std::filesystem::file_size(path) /
           (dims * sizeof(T) + sizeof(unsigned));
  }
}

unsigned getCutSize(unsigned expectSize, unsigned npts) {
  if (expectSize > npts) {
    std::cout << std::format("WARNING: Expected cut size: {} greater than npts "
                             "{} of dataset, Using npts as cut size ",
                             expectSize, npts)
              << std::endl;
    return npts;
  } else {
    return expectSize;
  }
}

unsigned readDims(const std::string &path, bool isBin = true) {
  unsigned dims{0};
  std::fstream reader = openBinaryFile(path, std::ios::in);
  if (isBin) {
    reader.seekg(sizeof(unsigned), std::ios::beg);
  }
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));
  return dims;
}

template <typename T> std::unique_ptr<T[]> readBin(const std::string &path) {
  std::fstream reader = openBinaryFile(path, std::ios::in);

  unsigned npts{0}, dims{0};

  reader.read(reinterpret_cast<char *>(&npts), sizeof(unsigned));
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));

  std::unique_ptr<T[]> data = std::make_unique<T[]>(npts * dims);
  reader.read(reinterpret_cast<char *>(data.get()), sizeof(T) * npts * dims);
  reader.close();
  return data;
}

template <typename T> std::unique_ptr<T[]> readVec(const std::string &path) {
  std::fstream reader = openBinaryFile(path, std::ios::in);
  unsigned npts{0}, dims{0};

  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));
  // NOTE: compute npts using file_size
  npts =
      std::filesystem::file_size(path) / (sizeof(unsigned) + sizeof(T) * dims);

  unsigned bufSize{
      static_cast<unsigned int>((sizeof(T) * dims + sizeof(unsigned)) * npts)};

  reader.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> byteData = std::make_unique<char[]>(bufSize);
  reader.read(reinterpret_cast<char *>(byteData.get()), bufSize);

  std::unique_ptr<T[]> data = std::make_unique<T[]>(npts * dims);
  unsigned nodeSize{
      static_cast<unsigned int>((sizeof(T) * dims + sizeof(unsigned)))};

  for (unsigned i{0}; i < npts; ++i) {
    std::memcpy(data.get() + i * dims,
                byteData.get() + i * nodeSize + sizeof(unsigned),
                sizeof(T) * dims);
  }

  return data;
}

template <typename T>
void writeBin(const std::string &path, const std::unique_ptr<T[]> data,
              const unsigned npts, const unsigned dims) {
  std::fstream writer = openBinaryFile(path, std::ios::out);
  writer.write(reinterpret_cast<const char *>(&npts), sizeof(unsigned));
  writer.write(reinterpret_cast<const char *>(&dims), sizeof(unsigned));
  writer.write(reinterpret_cast<const char *>(data.get()),
               sizeof(T) * npts * dims);
  writer.flush();
  writer.close();
}

template <typename T>
void writeVec(const std::string &path, const std::unique_ptr<T[]> data,
              const unsigned npts, const unsigned dims) {
  std::fstream writer = openBinaryFile(path, std::ios::out);
  unsigned bufSize = sizeof(T) * dims;
  for (unsigned i{0}; i < npts; ++i) {
    writer.write(reinterpret_cast<const char *>(&dims), sizeof(unsigned));
    writer.write(reinterpret_cast<char *>(data.get() + i * dims), bufSize);
  }
  writer.flush();
  writer.close();
}

template <typename T>
void transform(const std::string &sourcePath, const std::string &targetPath,
               bool isBin) {
  unsigned npts{0}, dims{0};
  loadMetaInfo<T>(sourcePath, npts, dims, isBin);
  std::cout << std::format("Source Data Info: npts[{}], dims[{}]", npts, dims)
            << std::endl;
  if (isBin) {
    writeVec<T>(targetPath, readBin<T>(sourcePath), npts, dims);
  } else {
    writeBin<T>(targetPath, readVec<T>(sourcePath), npts, dims);
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "ERROR: Argument Mismatch, Please Follow Usage" << std::endl;
    std::cout
        << "Usage: ./transform [source dataset path] [source dataset format] "
           "[target dataset path] [target dataset format] [data type]"
        << std::endl;

    exit(EXIT_FAILURE);
  }

  std::string sourcePath{};
  std::string sourceFormat{argv[2]};
  std::string targetPath{argv[3]};
  std::string targetFormat{argv[4]};
  std::string dataType{argv[5]};

  try {
    sourcePath = argv[1];
    checkFileExists(sourcePath);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  if (sourceFormat == targetFormat) {
    std::cout << std::format(
                     "Transform {} format to {} format? What Wrong With U???",
                     sourceFormat, targetFormat)
              << std::endl;
  } else {
    std::cout << std::format("Transform {} format to {} format", sourceFormat,
                             targetFormat)
              << std::endl;
  }

  bool isBin{sourceFormat == std::string("bin") ? true : false};

  if (sourceFormat != std::string("vecs") &&
      sourceFormat != std::string("bin")) {
    std::cerr << "ERROR: Input format does not meet requirements, Please Using "
                 "[bin] or [vecs] format ~"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  if (dataType == std::string("float")) {
    transform<float>(sourcePath, targetPath, isBin);
  } else if (dataType == std::string("uint")) {
    transform<uint>(sourcePath, targetPath, isBin);
  } else {
    std::cerr
        << "ERROR: Input DataType does not meet requirements, Please Using "
           "[float] or [uint] format ~"
        << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Transform Done!" << std::endl;
}

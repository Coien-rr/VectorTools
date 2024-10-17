#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <string>
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

unsigned getCutSize(unsigned expectSize, unsigned npts) {
  if (expectSize > npts) {
    std::cout << std::format(
                     "WARNING: Expected cut size: {} greater than npts "
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

std::unique_ptr<float[]> readBin(const std::string &path, const unsigned size) {
  std::fstream reader = openBinaryFile(path, std::ios::in);

  unsigned npts{0}, dims{0};

  reader.read(reinterpret_cast<char *>(&npts), sizeof(unsigned));
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));

  unsigned ss{getCutSize(size, npts)};

  std::unique_ptr<float[]> cutData = std::make_unique<float[]>(ss * dims);
  reader.read(reinterpret_cast<char *>(cutData.get()),
              sizeof(float) * ss * dims);
  reader.close();
  return cutData;
}

std::unique_ptr<char[]> readVec(const std::string &path, const unsigned size) {
  std::fstream reader = openBinaryFile(path, std::ios::in);
  unsigned npts{0}, dims{0};
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));
  // NOTE: compute npts using file_size
  auto totalFileSize{std::filesystem::file_size(path)};
  npts = totalFileSize / (sizeof(unsigned) + sizeof(float) * dims);

  unsigned ss{getCutSize(size, npts)};
  unsigned bufSize{static_cast<unsigned int>(
      (sizeof(float) * dims + sizeof(unsigned)) * ss)};

  reader.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> cutData = std::make_unique<char[]>(bufSize);
  reader.read(reinterpret_cast<char *>(cutData.get()), bufSize);
  return cutData;
}

void writeBin(const std::string &path, const std::unique_ptr<float[]> data,
              const unsigned npts, const unsigned dims) {
  std::fstream writer = openBinaryFile(path, std::ios::out);
  writer.write(reinterpret_cast<const char *>(&npts), sizeof(unsigned));
  writer.write(reinterpret_cast<const char *>(&dims), sizeof(unsigned));
  writer.write(reinterpret_cast<const char *>(data.get()),
               sizeof(float) * npts * dims);
  writer.flush();
  writer.close();
}

void writeVec(const std::string &path, const std::unique_ptr<char[]> data,
              const unsigned bufSize) {
  std::fstream writer = openBinaryFile(path, std::ios::out);
  writer.write(reinterpret_cast<const char *>(data.get()), bufSize);
  writer.flush();
  writer.close();
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "ERROR: Argument Mismatch, Please Follow Usage" << std::endl;
    std::cout << "Usage: ./slice [source dataset path] [source dataset format] "
                 "[target dataset path] [target dataset format] [size]"
              << std::endl;

    exit(EXIT_FAILURE);
  }

  unsigned counts{0};
  std::string sourcePath{};
  std::string sourceFormat{argv[2]};
  std::string targetPath{argv[3]};
  std::string targetFormat{argv[4]};

  try {
    sourcePath = argv[1];
    checkFileExists(sourcePath);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  try {
    counts = std::stoi(argv[5]);
  } catch (const std::invalid_argument &e) {
    std::cerr << "Error: Size Argument must be an integer." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << std::format(
                   "slice first [{}] vectors of [{}] and save it into [{}]",
                   counts, sourcePath, targetPath)
            << std::endl;

  if (sourceFormat == std::string("bin")) {
    unsigned dims{readDims(sourcePath)};
    writeBin(targetPath, readBin(sourcePath, counts), counts, dims);
  } else if (sourceFormat == std::string("vecs")) {
    unsigned dims{readDims(sourcePath, false)};
    writeVec(targetPath, readVec(sourcePath, counts),
             counts * (dims * sizeof(float) + sizeof(unsigned)));
    std::cout << std::format("Write {} bytes into {}",
                             counts * (dims * sizeof(float) + sizeof(unsigned)),
                             targetPath)
              << std::endl;
  } else {
    std::cerr << "ERROR: Input format does not meet requirements" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Cut Done!" << std::endl;
}

#include <algorithm> // std::iota
#include <chrono>    // std::chrono::system_clock
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random> // std::mt19937
#include <stdexcept>
#include <vector>

void loadMetaInfo(const std::filesystem::path &sourcePath, unsigned &npts,
                  unsigned &dims);

void checkFileSize(const std::filesystem::path &sourcePath,
                   const unsigned &npts, const unsigned &dims);

std::unique_ptr<float[]> loadMainData(const std::filesystem::path &sourcePath);

std::unique_ptr<float[]>
reorderData(const std::unique_ptr<float[]> originalData,
            const std::vector<unsigned> &ids, const unsigned &npts,
            const unsigned &dims);

void saveDestinationFile(const std::filesystem::path &destinationPath,
                         const std::unique_ptr<float[]> data,
                         const unsigned &npts, const unsigned &dims);

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Error Usage, Please Follow Usage!" << std::endl;
    std::cout << "./reorderData [source DataPath] [destination DataPath]"
              << std::endl;
    return -1;
  }

  std::filesystem::path sourcePath{argv[1]};
  std::filesystem::path destinationPath{argv[2]};
  unsigned npts{0}, dims{0};

  try {
    loadMetaInfo(sourcePath, npts, dims);
    checkFileSize(sourcePath, npts, dims);
  } catch (const std::exception &e) {
    std::cerr << "Runtime_ERROR: " << e.what() << std::endl;
  }

  std::vector<unsigned> ids(npts);
  std::iota(ids.begin(), ids.end(), 0);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 engine(seed);
  std::shuffle(ids.begin(), ids.end(), engine);

  try {
    std::unique_ptr<float[]> originalData = loadMainData(sourcePath);
    std::unique_ptr<float[]> data =
        reorderData(std::move(originalData), ids, npts, dims);
    saveDestinationFile(destinationPath, std::move(data), npts, dims);
    checkFileSize(destinationPath, npts, dims);
  } catch (const std::exception &e) {
    std::cerr << "Runtime_ERROR: " << e.what() << std::endl;
  }

  return 0;
}

void loadMetaInfo(const std::filesystem::path &sourcePath, unsigned &npts,
                  unsigned &dims) {
  std::ifstream reader(sourcePath, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Failed to open file: " + sourcePath.string());
  }
  reader.read((char *)&npts, sizeof(unsigned));
  reader.read((char *)&dims, sizeof(unsigned));
  reader.close();
}

void checkFileSize(const std::filesystem::path &sourcePath,
                   const unsigned &npts, const unsigned &dims) {
  std::uintmax_t actualFileSize{std::filesystem::file_size(sourcePath)};
  std::uintmax_t expectedFileSize =
      npts * dims * sizeof(float) + 2 * sizeof(unsigned);
  if (actualFileSize != expectedFileSize) {
    throw std::runtime_error(
        "File size mismatch: Expected " + std::to_string(expectedFileSize) +
        " bytes, but got " + std::to_string(actualFileSize) +
        " bytes for file: " + sourcePath.string());
  }
}

std::unique_ptr<float[]> loadMainData(const std::filesystem::path &sourcePath) {
  std::ifstream reader(sourcePath, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Failed to open file: " + sourcePath.string());
  }

  unsigned npts{0}, dims{0};
  reader.read((char *)&npts, sizeof(unsigned));
  reader.read((char *)&dims, sizeof(unsigned));

  std::unique_ptr<float[]> buf =
      std::make_unique<float[]>(npts * dims * sizeof(float));

  reader.read((char *)buf.get(), npts * dims * sizeof(float));

  reader.close();
  return buf;
}

std::unique_ptr<float[]>
reorderData(const std::unique_ptr<float[]> originalData,
            const std::vector<unsigned> &ids, const unsigned &npts,
            const unsigned &dims) {
  std::unique_ptr<float[]> newData =
      std::make_unique<float[]>(npts * dims * sizeof(float));

  for (unsigned i{0}; i < npts; ++i) {
    unsigned id = ids.at(i);
    std::copy(originalData.get() + i * dims,
              originalData.get() + (i + 1) * dims, newData.get() + id * dims);
  }

  return newData;
}

void saveDestinationFile(const std::filesystem::path &destinationPath,
                         const std::unique_ptr<float[]> data,
                         const unsigned &npts, const unsigned &dims) {
  std::ofstream writer(destinationPath, std::ios::binary);
  if (!writer.is_open()) {
    throw std::runtime_error("Failed to open file: " +
                             destinationPath.string());
  }
  unsigned writeLen = npts * dims * sizeof(float);
  writer.write((char *)&npts, sizeof(unsigned));
  writer.write((char *)&dims, sizeof(unsigned));
  writer.write((char *)data.get(), writeLen);
}

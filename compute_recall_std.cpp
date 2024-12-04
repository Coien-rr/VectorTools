#include <sys/types.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

void checkFileExists(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("File not found: " + path);
  }
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

void loadResultBin(const std::string &path, uint32_t &npts, uint32_t &dims,
                   uint32_t *data, float *resDists) {
  std::fstream reader = openBinaryFile(path, std::ios::in);

  reader.read(reinterpret_cast<char *>(&npts), sizeof(unsigned));
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));

  reader.read(reinterpret_cast<char *>(data), sizeof(float) * npts * dims);
  reader.close();
}

std::unique_ptr<uint32_t[]> readBin(const std::string &path) {
  std::fstream reader = openBinaryFile(path, std::ios::in);

  unsigned npts{0}, dims{0};

  reader.read(reinterpret_cast<char *>(&npts), sizeof(unsigned));
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));

  std::unique_ptr<uint32_t[]> data = std::make_unique<uint32_t[]>(npts * dims);
  reader.read(reinterpret_cast<char *>(data.get()),
              sizeof(float) * npts * dims);
  reader.close();
  return data;
}

std::unique_ptr<char[]> readVec(const std::string &path, const unsigned size) {
  std::fstream reader = openBinaryFile(path, std::ios::in);
  unsigned npts{0}, dims{0};
  reader.read(reinterpret_cast<char *>(&dims), sizeof(unsigned));
  // NOTE: compute npts using file_size
  auto totalFileSize{std::filesystem::file_size(path)};
  npts = totalFileSize / (sizeof(unsigned) + sizeof(float) * dims);

  unsigned bufSize{static_cast<unsigned int>(
      (sizeof(float) * dims + sizeof(unsigned)) * npts)};

  reader.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> data = std::make_unique<char[]>(bufSize);
  reader.read(reinterpret_cast<char *>(data.get()), bufSize);
  return data;
}

double calculateSingleRecall(uint32_t *gtIDs, float *gtDists, uint32_t gtDims,
                             uint32_t *resIDs, uint32_t resDims,
                             uint32_t topK) {
  std::set<uint32_t> gt, res;

  //  NOTE: Include all ground truth neighbors tied at the topK distance.
  uint32_t gtTieBreaker = topK;
  if (gtDists != nullptr) {
    gtTieBreaker = topK - 1;
    while (gtTieBreaker < gtDims &&
           gtDists[gtTieBreaker] == gtDists[topK - 1]) {
      ++gtTieBreaker;
    }
  }

  gt.insert(gtIDs, gtIDs + gtTieBreaker);
  res.insert(resIDs, resIDs + topK);

  unsigned curRecall = 0;
  for (auto id : res) {
    if (gt.find(id) != gt.end()) {
      ++curRecall;
    }
  }

  return static_cast<double>(curRecall) / topK;
}

double calculateTotalRecall(uint32_t numQueries, uint32_t *gtIDs,
                            float *gtDists, uint32_t gtDims, uint32_t *resIDs,
                            uint32_t resDims, uint32_t topK) {
  double totalRecall = 0;
  for (unsigned q{0}; q < numQueries; ++q) {
    std::set<uint32_t> gt, res;
    uint32_t *gtVec = gtIDs + q * gtDims;
    uint32_t *resVec = resIDs + q * resDims;
    float *gtDistVec = gtDists == nullptr ? nullptr : gtDists + q * gtDims;

    totalRecall +=
        calculateSingleRecall(gtVec, gtDistVec, gtDims, resVec, resDims, topK);
  }
  return totalRecall / numQueries;
}

std::vector<double> calculateRecallPerQuery(uint32_t numQueries,
                                            uint32_t *gtIDs, float *gtDists,
                                            uint32_t gtDims, uint32_t *resIDs,
                                            uint32_t resDims, uint32_t topK) {
  std::vector<double> recallList(numQueries);
  for (unsigned q{0}; q < numQueries; ++q) {
    uint32_t *gtVec = gtIDs + q * gtDims;
    uint32_t *resVec = resIDs + q * resDims;
    float *gtDistVec = gtDists == nullptr ? nullptr : gtDists + q * gtDims;

    recallList.at(q) =
        calculateSingleRecall(gtVec, gtDistVec, gtDims, resVec, resDims, topK);
  }
  return recallList;
}

double computeRecallSTD(const std::vector<double> &recallLists) {
  unsigned numQueries = recallLists.size();
  double mean =
      std::accumulate(recallLists.begin(), recallLists.end(), 0.0) / numQueries;
  double varience = 0;
  for (auto recall : recallLists) {
    varience += std::pow((recall - mean), 2);
  }
  varience /= numQueries;
  return std::sqrt(varience);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "ERROR: Argument Mismatch, Please Follow Usage" << std::endl;
    std::cout << "Usage: ./compute_recall_std "
                 "[result file path] [result format] [gt file path] [gt format]"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string resultPath{argv[1]};
  std::string resultFormat{argv[2]};
  std::string gtPath{argv[3]};
  std::string gtFormat{argv[4]};

  try {
    checkFileExists(resultPath);
    checkFileExists(gtPath);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << std::format("Read ResultInfo From {} To Compute STD Of Recall",
                           resultPath)
            << std::endl;

  std::cout << std::format("GroundTruth File is :{}", gtPath) << std::endl;

  if (resultFormat == "bin" && gtFormat == "bin") {
    std::cout << "bin" << std::endl;
    unsigned resQueries, gtQueries, resDims, gtDims;
    uint32_t *resIDs = nullptr, *gtIDs = nullptr;
    loadResultBin(resultPath, resQueries, resDims, resIDs, nullptr);
    loadResultBin(gtPath, gtQueries, gtDims, gtIDs, nullptr);
    double stdRecall = computeRecallSTD(calculateRecallPerQuery(
        resQueries, gtIDs, nullptr, gtDims, resIDs, resDims, 50));
    std::cout << stdRecall << std::endl;
  } else if (resultFormat == "vecs") {
    std::cout << "vecs" << std::endl;
  } else {
    std::cerr << "ERROR: Input format does not meet requirements" << std::endl;
    std::cout << "./compute_recall_std just support [vecs] or [bin] format now!"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

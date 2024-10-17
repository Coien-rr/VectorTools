#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

template <typename T, typename D>
std::unique_ptr<T[]> generateRandomDataByDistribution(const unsigned npts,
                                                      const unsigned dims,
                                                      D distribution) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::unique_ptr<T[]> res = std::make_unique<T[]>(npts * dims);
  for (unsigned i{0}; i < npts; ++i) {
    for (unsigned j{0}; j < dims; ++j) {
      *(res.get() + i * dims + j) = distribution(gen);
      // res.get()[i * dims + j] = distribution(gen);
    }
  }

  return std::move(res);
}

template <typename T>
std::unique_ptr<T[]> generateRandomData(const unsigned npts,
                                        const unsigned dims,
                                        const std::string_view distribution) {
  if (distribution == std::string("normal")) {
    // TODO: ask for mean && stddev for normal distribution
    if (std::is_same<float, T>::value) {
      std::normal_distribution<> dist(0, 1);
      return generateRandomDataByDistribution<T>(npts, dims, dist);
    } else {
      throw std::runtime_error("normal distribution only supports float now");
    }
  } else if (distribution == std::string("uniform")) {
    if (std::is_same<float, T>::value) {
      return generateRandomDataByDistribution<T>(
          npts, dims, std::uniform_real_distribution<>(-1, 1));
    } else if (std::is_same<int, T>::value) {
      return generateRandomDataByDistribution<T>(
          npts, dims, std::uniform_int_distribution<>(0, dims * 2));
    }
  } else {
    throw std::runtime_error(
        "Provided Distribution Error! "
        "Only support normal or uniform distribution now");
  }
  return nullptr;
}

void writeBin(const std::string &path, const std::unique_ptr<float[]> data,
              const unsigned npts, const unsigned dims) {
  std::ofstream writer(path, std::ios::binary);
  writer.write(reinterpret_cast<const char *>(&npts), sizeof(unsigned));
  writer.write(reinterpret_cast<const char *>(&dims), sizeof(unsigned));
  writer.write(reinterpret_cast<const char *>(data.get()),
               sizeof(float) * npts * dims);
  writer.flush();
  writer.close();
}

int main(int argc, char **argv) {
  if (argc < 6 || argc > 7) {
    std::cerr << "ERROR: Argument Mismatch, Please Follow Usage" << std::endl;
    std::cout << "Usage: ./generateRandomDataSet [npts] "
                 "[dims] [type] [format] [save_path] "
                 "[distribution(optional)]"
              << std::endl;
    return -1;
  }

  unsigned npts{0};
  unsigned dims{0};

  const std::string data_type{argv[3]};
  const std::string save_format{argv[4]};
  const std::string save_path{argv[5]};

  // NOTE: Using normal distribution as default
  std::string distribution{"normal"};
  if (argc == 7) {
    distribution.assign(argv[6]);
  }

  try {
    npts = std::stoi(argv[1]);
    dims = std::stoi(argv[2]);
  } catch (const std::invalid_argument &e) {
    std::cerr << "Error: [npts] && [dims] Argument must be an integer."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << std::format(
                   "Using [{}] Distribution Generate Random Data With "
                   "Info: npts[{}], dims[{}]",
                   distribution, npts, dims)
            << std::endl;

  try {
    if (data_type == std::string("float")) {
      writeBin(save_path, generateRandomData<float>(npts, dims, distribution),
               npts, dims);
    } else if (data_type == std::string("int")) {
      // generateRandomData<int>(npts, dims, distribution);
      throw std::runtime_error("Invalid Data_type, Only support float now");

    } else {
      throw std::runtime_error("Invalid Data_type, Only support float now");
    }
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

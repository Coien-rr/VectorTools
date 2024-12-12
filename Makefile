# Compiler flags
CXXFLAGS = -std=c++20 -Wall -O2

# Dynamically find all .cpp files in the current directory
SRC = $(wildcard *.cpp)

# Generate executable names by removing the .cpp extension
EXE = $(SRC:.cpp=)

# Default rule - build all executables
all: $(EXE)

# Rule to build each executable
%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean up build files
clean:
	rm -f $(EXE)

.PHONY: all clean

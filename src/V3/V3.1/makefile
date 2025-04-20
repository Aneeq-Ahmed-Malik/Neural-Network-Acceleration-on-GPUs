NVCC = nvcc                      # CUDA compiler
NVCCFLAGS = -O2 -arch=sm_86 -pg  # CUDA compiler flags (updated for Ampere architecture)
EXE = nn.exe                     # Executable name
SRC = nn.cu                      # CUDA source file

# Default target
all: $(EXE) run

# Link to create executable
$(EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(EXE) $(SRC) -lm

# Run the executable
run: $(EXE)
	./$(EXE)

# Clean up
clean:
	rm -f $(EXE) *.prof  # Clean executable and profiling files
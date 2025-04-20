CC = gcc
CFLAGS = -Wall -O2 -pg  # Added -pg for profiling
EXE = nn.exe
SRC = nn.c

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

profile: $(EXE)
	./$(EXE)  # Generates gmon.out
	gprof $(EXE) gmon.out > gprof_nn.txt  # Analyzes profiling data

clean:
	rm -f $(EXE) gmon.out gprof_nn.txt  callgraph.png callgraph.dot # Updated to clean profiling files

callgraph: profile
	python3 gprof2dot.py -s -n 2 -e 2 < gprof_nn.txt > callgraph.dot
	dot -Tpng callgraph.dot -o callgraph.png
.Phony: all debug

all:
	nvcc -std=c++11 main.cu mem.cpp -o renderer

debug:
	nvcc -g -G -std=c++11 main.cu mem.cpp -o renderer

src=main.cu

.Phony: all fast debug

all : ${src}
	clang++ -std=c++11 $^ -o ~/bin/render

fast : ${src}
	clang++ -std=c++11 $^ -O3 -o ~/bin/render

debug : ${src}
	clang++ -g -std=c++11 $^ -o ~/bin/render

gpu : ${src}
	nvcc -std=c++11 $^ -o render

# a.out : main.cpp
#     clang++ -std=c++11 $^ -o $@


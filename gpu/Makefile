src=main.cpp

.Phony: all debug

all : ${src}
	clang++ -std=c++11 $^ -O3 -o ~/bin/render

debug : ${src}
	clang++ -g -std=c++11 $^ -o ~/bin/render

# a.out : main.cpp
#     clang++ -std=c++11 $^ -o $@


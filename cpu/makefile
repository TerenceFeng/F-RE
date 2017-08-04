FLAGS		 = -w -g -std=c++11
OBJECTS		= Object.cpp Grid.cpp ply.cpp BBox.cpp
MODELS		= Light.cpp Material.cpp BRDF.cpp
UTILITIES	= sampler.cpp RGBColor.cpp ShadeRec.cpp Utilities.cpp World.cpp camera.cpp
TARGET		= renderer

default: clean renderer

test: clean $(TARGET)
	time ./$(TARGET)
	open result.ppm

$(TARGET):
	g++ main.cpp $(OBJECTS) $(MODELS) $(UTILITIES) $(FLAGS) -o $@

clean:
	rm -f renderer

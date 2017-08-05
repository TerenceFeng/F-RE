RELEASE		= -w -std=c++14 -O2
DEBUG		= -std=c++14 -g
OBJECTS		= object/BBox.cpp
MODELS		= Light.cpp Material.cpp BRDF.cpp
UTILITIES	= sampler.cpp RGBColor.cpp ShadeRec.cpp Utilities.cpp camera.cpp
TARGET		= renderer

default: clean release

test: clean $(TARGET)
	time ./$(TARGET)
	open result.ppm

release:
	g++ main.cpp $(OBJECTS) $(MODELS) $(UTILITIES) $(RELEASE) -o $(TARGET)

debug: clean_debug
	g++ main.cpp $(OBJECTS) $(MODELS) $(UTILITIES) $(DEBUG) -o debug

clean:
	rm -f renderer

clean_debug:
	rm -f debug

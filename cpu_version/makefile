FLAGS		 = -w -g -std=c++11
OBJECTS		= GeometricObject.cpp Grid.cpp ply.cpp BBox.cpp
MODELS		= Light.cpp Material.cpp BRDF.cpp
UTILITIES	= sampler.cpp RGBColor.cpp ShadeRec.cpp Utilities.cpp World.cpp Display.cpp camera.cpp

all: clean renderer

renderer:
	g++ jnf_renderer.cpp $(OBJECTS) $(MODELS) $(UTILITIES) $(FLAGS) -o $@

clean:
	rm -f renderer

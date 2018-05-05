CXXFLAGS = -w -Wall -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0

dash: dash.cpp functions.cpp
	g++ -o $@ $^ $(CXXFLAGS) 
clean: 
	rm -f dash *.o

CXX = g++
LIB_HOME = flowstar-toolbox
LIBS = -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -lpthread

#ACTION REQUIRED: Change the paths to the current user profile
CFLAGS = -I . -I/home/sundaran.sukanth/cartpole/flowstar-toolbox -I/apps/lib/mpfr/3.1.1/include \
-I/apps/lib/gmp/6.1.2/include -I/apps/gsl/2.6/include -I/apps/glpk/4.55/include -g -O3 -std=c++11

#ACTION REQUIRED: Change the paths to the current user profile
LINK_FLAGS = -g -L/home/sundaran.sukanth/cartpole/flowstar-toolbox \
-L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib -L/apps/gsl/2.6/lib \
-L/apps/glpk/4.55/lib -L/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 \
-L/apps/compilers/intel/2020/0.166/mkl/lib/intel64 \
-Wl,-rpath=/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 \
-Wl,-rpath=/apps/compilers/intel/2020/0.166/mkl/lib/intel64


OBJS = NeuralNetwork.o BernsteinPoly.o Neuron.o PolarSetting.o System.o Specification.o Polar.o

all: load_modules polar_lib polar_tool
# all: benchmark6_tanh

load_modules:
	source /etc/profile.d/modules.sh; \
        module load libgmp/6.1.2 gsl libmpfr/3.1.1 glpk intel/2020.0.166


polar_tool: Polar_tool.o $(OBJS)
	g++ -O3 -w $(LINK_FLAGS) -L./ -o $@ $^ $(LIBS)

polar_lib: $(OBJS)
	ar rcs libpolar.a $^

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o libpolar.a polar_tool
	

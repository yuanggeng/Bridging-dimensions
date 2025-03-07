CXX = g++
POLAR_HOME = /home/sundaran.sukanth/mountain_car/POLAR
FLOWSTAR_HOME = /home/sundaran.sukanth/mountain_car/flowstar-toolbox
#POLAR_HOME = POLAR
#FLOWSTAR_HOME = flowstar-toolbox
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -pthread

# Add the RPATH options to help the linker find the Intel libraries
LINK_FLAGS = -g -L$(FLOWSTAR_HOME) -L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib \
-L/apps/gsl/2.6/lib -L/apps/glpk/4.55/lib -L$(POLAR_HOME) \
-L/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 \
-L/apps/compilers/intel/2020/0.166/mkl/lib/intel64 \
-Wl,-rpath=/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 \
-Wl,-rpath=/apps/compilers/intel/2020/0.166/mkl/lib/intel64

CFLAGS = -I . -I$(FLOWSTAR_HOME) -I$(POLAR_HOME) -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/gsl/2.6/include -I/apps/glpk/4.55/include -g -O3 -std=c++11



all: mountain_car mountain_car_continuous mountain_car_noised_traj mountain_car_perfect_act mountain_car_perfect_traj

load_modules:
	source /etc/profile.d/modules.sh; \
	module load libgmp/6.1.2 gsl libmpfr/3.1.1 glpk

mountain_car: mountain_car.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

mountain_car_continuous: mountain_car_continuous.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

mountain_car_noised_traj: mountain_car_noised_traj.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

mountain_car_perfect_act: mountain_car_perfect_act.o 
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

mountain_car_perfect_traj: mountain_car_perfect_traj.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)
	
%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o mountain_car


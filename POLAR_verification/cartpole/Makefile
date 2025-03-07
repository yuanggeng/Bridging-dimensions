CXX = g++
#ACTION REQUIRED : Change the path variables to the current user.
POLAR_HOME = /home/sundaran.sukanth/cartpole/POLAR
FLOWSTAR_HOME = /home/sundaran.sukanth/cartpole/flowstar-toolbox


LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -pthread


LINK_FLAGS = -g -L$(FLOWSTAR_HOME) -L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib \
-L/apps/gsl/2.6/lib -L/apps/glpk/4.55/lib -L$(POLAR_HOME) \
-L/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 \
-L/apps/compilers/intel/2020/0.166/mkl/lib/intel64 \
-Wl,-rpath=/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 \
-Wl,-rpath=/apps/compilers/intel/2020/0.166/mkl/lib/intel64

CFLAGS = -I . -I$(FLOWSTAR_HOME) -I$(POLAR_HOME) -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/gsl/2.6/include -I/apps/glpk/4.55/include -g -O3 -std=c++11


all: double_pendulum_less double_pendulum_more double_pendulum_noised_traj double_pendulum_perfect_act double_pendulum_perfect_traj

load_modules:
	source /etc/profile.d/modules.sh; \
	module load libgmp/6.1.2 gsl libmpfr/3.1.1 glpk

double_pendulum_less: double_pendulum_less.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)
	
double_pendulum_more: double_pendulum_more.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

double_pendulum_noised_traj: double_pendulum_noised_traj.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

double_pendulum_perfect_act: double_pendulum_perfect_act.o 
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

double_pendulum_perfect_traj: double_pendulum_perfect_traj.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o double_pendulum_less double_pendulum_more double_pendulum_noised_traj double_pendulum_perfect_act double_pendulum_perfect_traj 

CXX = g++
POLAR_HOME = /home/sundaran.sukanth/single_pendulum/POLAR
FLOWSTAR_HOME = /home/sundaran.sukanth/single_pendulum/flowstar-toolbox
#POLAR_HOME = POLAR
#FLOWSTAR_HOME = flowstar-toolbox
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk -pthread
CFLAGS = -I . -I$(FLOWSTAR_HOME) -I$(POLAR_HOME) -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/gsl/2.6/include -I/apps/glpk/4.55/include -g -O3 -std=c++11
LINK_FLAGS = -g -L$(FLOWSTAR_HOME) -L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib -L/apps/gsl/2.6/lib -L/apps/glpk/4.55/lib -L$(POLAR_HOME)


#CFLAGS = -I . -I/home/sundaran.sukanth/single_pendulum/flowstar-toolbox -I/home/sundaran.sukanth/single_pendulum/POLAR -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/gsl/2.6/include -I/apps/glpk/4.55/include -g -O3 -std=c++11

#LINK_FLAGS = -g -L/home/sundaran.sukanth/single_pendulum/flowstar-toolbox -L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib -L/apps/gsl/2.6/lib -L/apps/glpk/4.55/lib -L/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 -L/apps/compilers/intel/2020/$


#CFLAGS = -I . -I/home/sundaran.sukanth/single_pendulum/flowstar-toolbox CFLAGS = -I . -I/home/sundaran.sukanth/single_pendulum/flowstar-toolbox -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/gs -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/gsl/2.6/include -I/apps/glpk/4.55/include -g -O3 -std=c++11

#LINK_FLAGS = -g -L/home/sundaran.sukanth/single_pendulum/flowstar-toolbox -L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib -L/apps/gsl/2.6/lib -L/apps/glpk/4.55/lib -L/apps/compilers/intel/2020/0.166/compilers_and_libraries/linux/lib64 -L/apps/compilers/intel/2020/$



#CFLAGS = -I . -g -O3 -std=c++11 -I$(HPC_GSL_INC) -I$(HPC_LIBMPFR_INC) -I$(HPC_GLPK_INC)
#LINK_FLAGS = -g -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L$(HPC_GSL_LIB) -L$(HPC_LIBMPFR_LIB) -L$(HPC_GLPK_LIB)


all: single_pendulum single_pendulum_noised_traj single_pendulum_perfect_action single_pendulum_perfect_traj

load_modules:
	source /etc/profile.d/modules.sh; \
	module load libgmp/6.1.2 gsl libmpfr/3.1.1 glpk

single_pendulum: single_pendulum.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

#single_pendulum_noised_traj: single_pendulum_noised_traj.o
#	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

#single_pendulum_perfect_action: single_pendulum_perfect_action.o
#	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

#single_pendulum_perfect_traj: single_pendulum_perfect_traj.o
#	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o single_pendulum



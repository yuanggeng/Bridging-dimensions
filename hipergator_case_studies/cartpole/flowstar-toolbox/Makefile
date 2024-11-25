CXX = g++
LIBS = -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk

CFLAGS = -I . -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/gsl/2.6/include -I/apps/glpk/4.55/include -g -O3 -std=c++11

LINK_FLAGS = -g -L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib -L/apps/gsl/2.6/lib -L/apps/glpk/4.55/lib



#CFLAGS = -I . -I/apps/lib/mpfr/3.1.1/include -I/apps/lib/gmp/6.1.2/include -I/apps/lib/gsl/2.6/include -g -O3 -std=c++11

#LINK_FLAGS = -g -L/apps/lib/mpfr/3.1.1/lib -L/apps/lib/gmp/6.1.2/lib -L/apps/lib/gsl/2.6/lib -L$(HPC_GLPK_LIB)



#CFLAGS = -I . -I$(HPC_GSL_INC) -I/apps/lib/mpfr/3.1.1/include -I$(HPC_MPFR_INC) -I$(HPC_GLPK_INC) -I$(HPC_GMP_INC) -g -O3 -std=c++11
#CFLAGS = -I . -I$(HPC_GSL_INC) -I$(HPC_MPFR_INC) -I$(HPC_GLPK_INC) -I$(HPC_GMP_INC) -g -O3 -std=c++11
#LINK_FLAGS = -g -L$(HPC_GSL_LIB) -L$(HPC_MPFR_LIB) -L$(HPC_GLPK_LIB) -L$(HPC_GMP_LIB)
OBJS = Interval.o Variables.o settings.o Matrix.o Geometry.o Constraints.o Continuous.o Discrete.o Hybrid.o

all: load_modules lib


load_modules:
	source /etc/profile.d/modules.sh; \
	module load libgmp/6.1.2 gsl libmpfr/3.1.1 glpk


lib: $(OBJS) lex.yy.o modelParser.tab.o
	ar rcs libflowstar.a $^

%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<

modelParser.tab.c: modelParser.y
	bison -d -v modelParser.y

lex.yy.c: modelLexer.l modelParser.tab.c
	flex modelLexer.l

clean: 
	rm -f *.o libflowstar.a *~ libflowstar.a *~ modelParser.tab.c modelParser.tab.h modelParser.output lex.yy.c



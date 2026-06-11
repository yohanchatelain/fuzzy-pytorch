SHELL=/bin/sh
BENCHMARK=sp
BENCHMARKU=SP
NBPPARAMS=-DNBPPARAMS="npbparams-${CLASS}.hpp"

include ../config/make.def

OBJS =	sp_${CLASS}.o \
	${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} ${NBPPARAMS}

sp_${CLASS}.o:		sp.cpp npbparams-${CLASS}.hpp
	${CCOMPILE} sp.cpp -o sp_${CLASS}.o  ${NBPPARAMS}

clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams-${CLASS}.hpp core

SHELL=/bin/sh
BENCHMARK=ep
BENCHMARKU=EP
NBPPARAMS=-DNBPPARAMS="npbparams-${CLASS}.hpp"

include ../config/make.def

OBJS = ep_${CLASS}.o ${COMMON}/c_print_results.o ${COMMON}/c_${RAND}.o \
       ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} ${NBPPARAMS}

ep_${CLASS}.o:		ep.cpp npbparams-${CLASS}.hpp
	${CCOMPILE} ep.cpp -o ep_${CLASS}.o ${NBPPARAMS}

clean:
	- rm -f *.o *~ 
	- rm -f npbparams-${CLASS}.hpp core

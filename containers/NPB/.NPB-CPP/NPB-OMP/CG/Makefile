SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG
NBPPARAMS=-DNBPPARAMS="npbparams-${CLASS}.hpp"

include ../config/make.def

OBJS = cg_${CLASS}.o ${COMMON}/c_print_results.o  \
       ${COMMON}/c_${RAND}.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}  ${NBPPARAMS}

cg_${CLASS}.o:             cg.cpp npbparams-${CLASS}.hpp
	${CCOMPILE} cg.cpp -o cg_${CLASS}.o ${NBPPARAMS}

clean:
	- rm -f *.o *~ 
	- rm -f npbparams-${CLASS}.hpp core

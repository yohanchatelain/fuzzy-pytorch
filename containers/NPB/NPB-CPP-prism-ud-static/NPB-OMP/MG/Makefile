SHELL=/bin/sh
BENCHMARK=mg
BENCHMARKU=MG
NBPPARAMS=-DNBPPARAMS="npbparams-${CLASS}.hpp"

include ../config/make.def

OBJS = mg_${CLASS}.o ${COMMON}/c_print_results.o  \
       ${COMMON}/c_${RAND}.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} ${NBPPARAMS}

mg_${CLASS}.o:		mg.cpp npbparams-${CLASS}.hpp
	${CCOMPILE} mg.cpp -o mg_${CLASS}.o ${NBPPARAMS}

clean:
	- rm -f *.o *~ 
	- rm -f npbparams-${CLASS}.hpp core

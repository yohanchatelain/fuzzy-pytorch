SHELL=/bin/sh
BENCHMARK=lu
BENCHMARKU=LU
NBPPARAMS=-DNBPPARAMS="npbparams-${CLASS}.hpp"

include ../config/make.def

OBJS = lu_${CLASS}.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} ${NBPPARAMS}

lu_${CLASS}.o:		lu.cpp npbparams-${CLASS}.hpp
	${CCOMPILE} lu.cpp -o lu_${CLASS}.o ${NBPPARAMS}

clean:
	- /bin/rm -f npbparams-${CLASS}.hpp
	- /bin/rm -f *.o *~

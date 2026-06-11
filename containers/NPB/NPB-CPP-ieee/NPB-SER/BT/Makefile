SHELL=/bin/sh
BENCHMARK=bt
BENCHMARKU=BT
NBPPARAMS=-DNBPPARAMS="npbparams-${CLASS}.hpp"

include ../config/make.def

OBJS =	bt_${CLASS}.o \
	${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} ${NBPPARAMS}

bt_${CLASS}.o:             bt.cpp npbparams-${CLASS}.hpp
	${CCOMPILE} bt.cpp -o bt_${CLASS}.o ${NBPPARAMS}

clean:
	- rm -f *.o *~ mputil*
	- rm -f  npbparams-${CLASS}.hpp core

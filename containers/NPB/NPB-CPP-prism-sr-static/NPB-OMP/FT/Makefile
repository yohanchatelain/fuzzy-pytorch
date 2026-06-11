SHELL=/bin/sh
BENCHMARK=ft
BENCHMARKU=FT
NBPPARAMS=-DNBPPARAMS="npbparams-${CLASS}.hpp"

include ../config/make.def

OBJS = ft_${CLASS}.o ${COMMON}/c_${RAND}.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/c_wtime.o #../omp-prof.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB} ${NBPPARAMS}

ft_${CLASS}.o:             ft.cpp npbparams-${CLASS}.hpp
	${CCOMPILE} ft.cpp -o ft_${CLASS}.o ${NBPPARAMS}

clean:
	- rm -f *.o *~ mputil*
	- rm -f ft npbparams-${CLASS}.hpp core

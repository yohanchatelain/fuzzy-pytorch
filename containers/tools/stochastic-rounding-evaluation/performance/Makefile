CC=clang++
# We now use "-mfpmath=sse -msse2" to avoid x87 80-bit arithmetic.
WCFLAGS=-Wall -Wextra # Warnings
MCFLAGS=-march=native -mfpmath=sse # -mfma -msse2 included in -march=native.
CFLAGS=$(WCFLAGS) $(MCFLAGS) -fPIC
CFLAGS_LO=$(CFLAGS) -O0 # Use with switching of rounding modes.
CFLAGS_HO=$(CFLAGS) -O3 # Use with RN-only and MPFR version of algorithms.
CLIBS=-lgmp -lmpfr -lm
CMAKE_PATH=/opt/local/lib/
CMAKE_INCLUDE=/opt/local/include/
DEPS=stochrndhp.h stochrnddw.hpp
OBJ=stochrndhp.o stochrnddw_rn.o stochrnddw_ar.o
.PRECIOUS: %.o

PREFIX=$(if $(VIRTUAL_ENV),$(VIRTUAL_ENV),/usr/local/)
PREFIX_LIB=$(PREFIX)lib/
PREFIX_INC=$(PREFIX)include/

all: srlib test_performance test_validation

run_%: test_%
	./$<

srlib: $(OBJ)
	$(CC) $(CFLAGS_HO) -shared -fpic -o libsr.so $^ $(CLIBS)
	ar rcs libsr.a $^

test_%: test_%.o $(OBJ)
	$(CC) $(CFLAGS_HO) -L$(CMAKE_PATH) -o $@ $^ $(CLIBS)

stochrnddw_ar.o: stochrnddw_ar.cpp stochrnddw.hpp
	$(CC) $(CFLAGS_LO) -I$(CMAKE_INCLUDE) -c -o $@ $<

stochrnddw_rn.o: stochrnddw_rn.cpp stochrnddw.hpp
	$(CC) $(CFLAGS_HO) -I$(CMAKE_INCLUDE) -c -o $@ $<

stochrndhp.o: stochrndhp.cpp stochrndhp.h
	$(CC) $(CFLAGS_HO) -I$(CMAKE_INCLUDE) -c -o $@ $<

test_%.o: test_%.c $(DEPS)
	$(CC) $(CFLAGS_HO) -I$(CMAKE_INCLUDE) -c -o $@ $<

install: srlib
	cp *.hpp ${PREFIX_INC}
	cp libsr.so libsr.a ${PREFIX_LIB}


profile: test_performance
	valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes \
		./test_performance
	qcachegrind callgrind.out.*

query:
	gcc -Q -march=native --help=target

clean:
	rm -f *.o test_performance test_validation callgrind.out.* libsr.*

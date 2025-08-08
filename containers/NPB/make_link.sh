#!/bin/bash

# Create sys and common links
for i in NPB-CPP-{ieee,prism-sr-dynamic,prism-sr-static,prism-ud-dynamic,prism-ud-static,verificarlo} ; do
  for p in SER OMP ; do
    cd "$i"/NPB-$p || exit ;
    ln -s ../../.NPB-CPP/NPB-${p}/sys ;
    ln -s ../../.NPB-CPP/NPB-${p}/common ;
    cd ../.. ;
  done ;
done

# Create benchmark links
for i in NPB-CPP-{ieee,prism-sr-dynamic,prism-sr-static,prism-ud-dynamic,prism-ud-static,verificarlo} ; do
  for p in SER OMP; do
    cd "$i"/NPB-$p || exit ;
    for b in BT CG EP FT LU MG SP ; do
      cd "$b" || exit ;
      ln -s ../../../.NPB-CPP/NPB-${p}/${b}/${b,,}.cpp ;
      ln -s ../../../.NPB-CPP/NPB-${p}/${b}/Makefile ;
      cd .. ;
    done ;
    cd ../.. ;
  done ;
done

# Create benchmark links for cadna
i=NPB-CPP-cadna
for p in SER OMP; do
  cd "$i"/NPB-$p || exit ;
  for b in BT CG EP FT LU MG SP ; do
    cd "$b" || exit ;
    ln -s ../../../.NPB-CPP/NPB-${p}/${b}/${b,,}_cad.cpp ${b,,}.cpp ;
    ln -s ../../../.NPB-CPP/NPB-${p}/${b}/Makefile ;
    cd .. ;
  done ;
  cd ../.. ;
done

# Create benchmark links for sr
# Create benchmark links for sr
i=NPB-CPP-sr
for p in SER OMP; do
  cd "$i"/NPB-$p || exit ;
  for b in BT CG EP FT LU MG SP ; do
    cd $b || exit ;
    ln -s ../../../.NPB-CPP/NPB-${p}/${b}/${b,,}_sr.cpp ${b,,}.cpp ;
    ln -s ../../../.NPB-CPP/NPB-${p}/${b}/Makefile ;
    cd .. ;
  done ;
  cd ../.. ;
done

#!/bin/bash

# Remove all files in the output directory

for tool in $(ls -1d NPB-CPP-*); do
  make -C $tool/NPB-SER cleanall
  make -C $tool/NPB-OMP cleanall
  find . -type f -name '*.ll' -exec rm {} +
  find . -type f -name '*.o' -exec rm {} +
done

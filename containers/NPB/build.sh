#!/bin/bash

function build_tool() {
  rm -rf make_logs
  mkdir -p make_logs

  # Build setparams once to avoid concurrent writes
  make -C NPB-SER/sys clean
  make -C NPB-SER/sys

  parallel -t --progress --halt now,fail=1 --header : \
  "make --silent -C NPB-SER CLASS={class} {bench}  &> make_logs/make.{bench}.{class}.log" \
  ::: class S A   \
  ::: bench bt cg ep ft lu mg sp

  # Build setparams once to avoid concurrent writes
  make -C NPB-OMP/sys clean
  make -C NPB-OMP/sys

  parallel -t --progress --halt now,fail=1 --header : \
  "make --silent -C NPB-OMP CLASS={class} {bench}  &> make_logs/make.{bench}.{class}.log" \
  ::: class S A   \
  ::: bench bt cg ep ft lu mg sp

}

# Build NPB-CPP for each tool
for tool in NPB-CPP-*; do
  echo "Building NPB-CPP-SER for $tool"
  cd "$tool" || exit
  build_tool
  if ! build_tool; then
    echo "Error occurred while building $tool. Stopping."
    exit 1
  fi
  cd ..
done

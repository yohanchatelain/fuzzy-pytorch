#!/bin/bash

set -e

root=$(pwd)

CADNA_SRC_DIR=cadna_c-3.1.11
CADNA_TAR=${CADNA_SRC_DIR}.tar.gz
CADNA_TAR_LINK=https://www-pequan.lip6.fr/cadna/Download_Dir/$CADNA_TAR

function install_cadna {
  cd $root
  echo "Installing CADNA..."
  if [ ! -f $CADNA_TAR ]; then
    echo "${CADNA_TAR} not found, downloading..."
    wget --no-check-certificate -O $CADNA_TAR $CADNA_TAR_LINK
    tar -xzf $CADNA_TAR
  fi
  cd $CADNA_SRC_DIR
  ./configure
  make
  make install
  echo "CADNA installed."
}

SR_DIR=stochastic-rounding-evaluation

function install_sr {
  cd $root
  echo "Installing SR..."
  cd $SR_DIR
  make -C performance/
  make -C performance/ install
  echo "SR installed."
}

VERROU_SRC_DIR=valgrind-3.23.0+verrou-2.6.0
VERROU_TAR=valgrind-3.23.0_verrou-2.6.0.tar.gz
VERROU_TAR_LINK=https://github.com/edf-hpc/verrou/releases/download/v2.6.0/$VERROU_TAR

function install_verrou {
  cd $root
  echo "Installing Verrou..."
  if [ ! -f $VERROU_TAR ]; then
    echo "${VERROU_TAR} not found, downloading..."
    wget --no-check-certificate -O $VERROU_TAR $VERROU_TAR_LINK
    tar -xzf $VERROU_TAR -C .
  fi
  cd $VERROU_SRC_DIR
  ./autogen.sh
  ./configure --enable-only64bit
  make
  make install
  echo "Verrou installed."
}


VERIFICARLO_DIR=verificarlo-2.2.0
VERIFICARLO_TAR=v2.2.0.tar.gz
VERIFICARLO_TAR_LINK=https://github.com/verificarlo/verificarlo/archive/refs/tags/v2.2.0.tar.gz

PRISM_DIR=verificarlo-2.2.0/src/backends/prism
PRISM_TAR=v0.0.2.tar.gz
PRISM_TAR_LINK=https://github.com/yohanchatelain/prism/archive/refs/tags/v0.0.2.tar.gz

function install_verificarlo {
  cd $root
  echo "Installing Verificarlo..."
  if [ ! -f $VERIFICARLO_TAR ]; then
    echo "${VERIFICARLO_TAR} not found, downloading..."
    wget --no-check-certificate -O $VERIFICARLO_TAR $VERIFICARLO_TAR_LINK
  fi
  if [ ! -d $VERIFICARLO_DIR ]; then
    echo "${VERIFICARLO_DIR} not found, extracting..."
    tar -xzf $VERIFICARLO_TAR
  fi
  if [ ! -f $PRISM_TAR ]; then
    echo "${PRISM_TAR} not found, downloading..."
    wget --no-check-certificate -O $PRISM_TAR $PRISM_TAR_LINK
  fi
  if [ ! -d $PRISM_DIR ] || [ -z "$(ls -A $PRISM_DIR 2>/dev/null)" ]; then
    echo "${PRISM_DIR} not found, extracting..."
    tar -xzf $PRISM_TAR
    rmdir $PRISM_DIR
    mv prism-0.0.2 $PRISM_DIR
    # Patch to fix march flag
    sed -i "s/-march=native/-march=haswell/g" $PRISM_DIR/constants.bzl
    sed -i "s/-mtune=native/-mtune=haswell/g" $PRISM_DIR/constants.bzl
    sed '/^NATIVE_COPTS = \[/a\    "-maes",' $PRISM_DIR/constants.bzl
  fi
  cd $VERIFICARLO_DIR
  ./autogen.sh
  ./configure --without-flang
  make install-interflop-stdlib
  make
  make install
  echo "Verificarlo installed."
}



function install {
  install_cadna
  install_sr
  install_verrou
  install_verificarlo
}

install

#!/bin/bash

export VFC_BACKENDS="libinterflop_ieee.so"

VERBOSE=1
META_REPETITIONS=3

# Ensure VERBOSE is an integer
VERBOSE=${VERBOSE:-0}

# Check for required dependencies
command -v bc >/dev/null 2>&1 || { echo "Error: bc is required but not installed." >&2; exit 1; }
command -v parallel >/dev/null 2>&1 || { echo "Error: parallel is required but not installed." >&2; exit 1; }
command -v valgrind >/dev/null 2>&1 || { echo "Error: valgrind is required but not installed." >&2; exit 1; }

# Ensure tool, bench, class, and threads are provided as arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
  echo "Error: No tool, bench, class, or threads provided. Usage: $0 <tool> <bench> <class> <threads>"
  echo "Tools: cadna ieee prism_sr_static prism_ud_static prism_sr_dynamic prism_ud_dynamic sr verificarlo verrou_cestac verrou_sr"
  echo "Benchmarks: bt cg ep ft is lu mg sp"
  echo "Classes: S A B C D E"
  exit 1
fi

TOOL=$1
BENCH=$2
CLASS=$3
THREADS=$4

function run_command {
  local cmd=$1
  local threads=$2
  local total=0
  local sq_total=0
  
  if [ ${VERBOSE} -eq 1 ]; then
    echo -n "run command: $cmd | "
  fi
  
  start=$(date +%s.%N)
  OMP_NUM_THREADS=$threads eval $cmd &> /dev/null
  ret=$?
  end=$(date +%s.%N)
  if [ $ret -ne 0 ]; then
    echo "Error: command failed with return code $ret"
    exit 1
  fi
  duration=$(echo "$end - $start" | bc)
  echo "$duration"
  
}

export -f run_command
export VERBOSE

# CADNA


function run_cadna() {
  CADNA_SER_PATH=$PWD/NPB-CPP-cadna/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$CADNA_SER_PATH" ]; then
    echo "Error: CADNA path $CADNA_SER_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$CADNA_SER_PATH/bin/$bench.$class
  local results=results/cadna_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run cadna ${bench} ${class} with ${threads} threads"
  fi
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


# IEEE


function run_ieee() {
  IEEE_SER_PATH=$PWD/NPB-CPP-ieee/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$IEEE_SER_PATH" ]; then
    echo "Error: IEEE path $IEEE_SER_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$IEEE_SER_PATH/bin/$bench.$class
  local results=results/ieee_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run ieee ${bench} ${class} with ${threads} threads"
  fi
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


# PRISM SR STATIC

function run_prism_sr_static() {
  PRISM_SR_STATIC_PATH=$PWD/NPB-CPP-prism-sr-static/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$PRISM_SR_STATIC_PATH" ]; then
    echo "Error: PRISM SR STATIC path $PRISM_SR_STATIC_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$PRISM_SR_STATIC_PATH/bin/$bench.$class
  local results=results/prism_sr_static_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run prism_sr_static ${bench} ${class} with ${threads} threads"
  fi
  
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


# PRISM UD STATIC


function run_prism_ud_static() {
  PRISM_UD_STATIC_PATH=$PWD/NPB-CPP-prism-ud-static/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$PRISM_UD_STATIC_PATH" ]; then
    echo "Error: PRISM UD STATIC path $PRISM_UD_STATIC_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$PRISM_UD_STATIC_PATH/bin/$bench.$class
  local results=results/prism_ud_static_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run prism_ud_static ${bench} ${class} with ${threads} threads"
  fi
  
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


# PRISM SR DYNAMIC

function run_prism_sr_dynamic() {
  PRISM_SR_DYNAMIC_PATH=$PWD/NPB-CPP-prism-sr-dynamic/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$PRISM_SR_DYNAMIC_PATH" ]; then
    echo "Error: PRISM SR DYNAMIC path $PRISM_SR_DYNAMIC_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$PRISM_SR_DYNAMIC_PATH/bin/$bench.$class
  local results=results/prism_sr_dynamic_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run prism_sr_dynamic ${bench} ${class} with ${threads} threads"
  fi
  
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


# PRISM UD DYNAMIC


function run_prism_ud_dynamic() {
  PRISM_UD_DYNAMIC_PATH=$PWD/NPB-CPP-prism-ud-dynamic/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$PRISM_UD_DYNAMIC_PATH" ]; then
    echo "Error: PRISM UD DYNAMIC path $PRISM_UD_DYNAMIC_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$PRISM_UD_DYNAMIC_PATH/bin/$bench.$class
  local results=results/prism_ud_dynamic_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run prism_ud_dynamic ${bench} ${class} with ${threads} threads"
  fi
  
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


# SR


function run_sr() {
  SR_PATH=$PWD/NPB-CPP-sr/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$SR_PATH" ]; then
    echo "Error: SR path $SR_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$SR_PATH/bin/$bench.$class
  local results=results/sr_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run sr ${bench} ${class} with ${threads} threads"
  fi
  
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


# VERIFICARLO MCA BACKEND


function run_verificarlo() {
  VERIFICARLO_PATH=$PWD/NPB-CPP-verificarlo/NPB-OMP
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$VERIFICARLO_PATH" ]; then
    echo "Error: VERIFICARLO path $VERIFICARLO_PATH does not exist" >&2
    return 1
  fi
  
  local cmd=$VERIFICARLO_PATH/bin/$bench.$class
  local results=results/verificarlo_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run verificarlo ${bench} ${class} with ${threads} threads"
  fi
  
  export VFC_BACKENDS="libinterflop_mca_int.so -m rr"
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}



function run_verrou_cestac() {
  IEEE_SER_PATH=$PWD/NPB-CPP-ieee/NPB-OMP
  VERROU_PATH=$IEEE_SER_PATH
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$VERROU_PATH" ]; then
    echo "Error: VERROU path $VERROU_PATH does not exist" >&2
    return 1
  fi
  
  local cmd="valgrind --tool=verrou --rounding-mode=random ${VERROU_PATH}/bin/$bench.$class"
  local results=results/verrou_cestac_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run verrou ${bench} ${class} with ${threads} threads"
  fi
  
  parallel -j1 --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}


function run_verrou_sr() {
  IEEE_SER_PATH=$PWD/NPB-CPP-ieee/NPB-OMP
  VERROU_PATH=$IEEE_SER_PATH
  local bench=${1,,}
  local class=${2^^}
  local rep=$3
  local threads=$4
  
  # Check if path exists
  if [ ! -d "$VERROU_PATH" ]; then
    echo "Error: VERROU path $VERROU_PATH does not exist" >&2
    return 1
  fi
  
  local cmd="valgrind --tool=verrou --rounding-mode=average ${VERROU_PATH}/bin/$bench.$class"
  local results=results/verrou_sr_${bench}_${class}_${threads}_results
  
  # Create results directory
  mkdir -p "$results"
  
  if [ $VERBOSE -eq 1 ]; then
    echo  "run verrou ${bench} ${class} with ${threads} threads"
  fi
  
  parallel --files --results "$results" --header "run_command cmd thread" ::: cmd "${cmd}" ::: repetition $(seq $rep) ::: thread $threads
}

export -f run_cadna
export -f run_ieee
export -f run_prism_sr_static
export -f run_prism_ud_static
export -f run_prism_sr_dynamic
export -f run_prism_ud_dynamic
export -f run_sr
export -f run_verificarlo
export -f run_verrou_cestac
export -f run_verrou_sr

# parallel --header : run_{tool} {bench} {class} {rep} {threads} \
#     ::: tool cadna ieee prism_sr_static prism_ud_static prism_sr_dynamic prism_ud_dynamic sr verificarlo verrou_cestac verrou_sr \
#     ::: bench $BENCH \
#     ::: class $CLASS \
#     ::: rep $META_REPETITIONS \
#     ::: threads 2   4 8 16 32


run_${TOOL} ${BENCH} ${CLASS} ${META_REPETITIONS} ${THREADS}

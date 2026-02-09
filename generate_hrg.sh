#!/bin/sh

### Usage ###

# ./generate_hrg.sh 1024 128

# This generates 128 hyperbolic random graphs, each with 1024 nodes.


### Setup ###

# path to executable (hyperbolic random graph generator from https://github.com/chistopher/girgs)
bin_path=~/girgs/build/genhrg

# path to output directory
output_dir=./graphs/

###


n=$1
repetitions=$2
rmax=($repetitions - 1)

for r in $(seq 0 $rmax)
do
	outputfile=$output_dir"hrg_"$n"_"$r"_edges"
	$bin_path -n $n -edge 1 -file $outputfile -rseed $RANDOM -aseed $RANDOM -sseed $RANDOM
	sed -i '1s/^/% /' $outputfile".txt"
done		


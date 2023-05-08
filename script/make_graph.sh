#!/bin/sh

rm -rf graphs
mkdir graphs

for TUPLE in "sift1M 128 1000000" "gist1M 960 1000000" "crawl 300 1989995" "msong 420 994185" "deep1M 256 1000000" "glove-100 100 1183514"
do
	set -- $TUPLE
	DATASET=$1
	VEC_DIM=$2
	NUM_VEC=$3
    	DATASET_LOWER=$(echo ${DATASET} | tr "[:upper:]" "[:lower:]")

	rm -rf build_template
	./generate_template.sh && ./fill_parameters.sh 50 ${VEC_DIM} l2
    	./build_graph.sh datasets/${DATASET}/${DATASET_LOWER}_base.fvecs ${NUM_VEC} ${VEC_DIM} l2 
    	sudo mv bfsg.data ./graphs/bfsg_${DATASET}.data
    	sudo mv bfsg.graph ./graphs/bfsg_${DATASET}.graph
done

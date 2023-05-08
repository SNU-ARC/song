#!/bin/bash

rm -rf result
mkdir result
cd result 
mkdir song_whole_batch
mkdir song_single_batch
cd ..

# Single batch repeated once due to caching, and averaged
for BATCH_OPTION in "batch 0 3" "single 1 1"
do
	set -- $BATCH_OPTION
	PRINT_FILE=$1
	SET_SINGLE_BATCH=$2
	END=$3

	rm ./result/${PRINT_FILE}.txt

	for TUPLE in "sift1M 128 1000000" "gist1M 960 1000000" "crawl 300 1989995" "msong 420 994185" "glove-100 100 1183514" "deep1M 256 1000000"
	do
		set -- $TUPLE
		DATASET=$1
		VEC_DIM=$2
		NUM_VEC=$3

		DATASET_LOWER=$(echo ${DATASET} | tr "[:upper:]" "[:lower:]")

		rm bfsg.data
		rm bfsg.graph
		ln -s ./graphs/bfsg_${DATASET}.data bfsg.data
		ln -s ./graphs/bfsg_${DATASET}.graph bfsg.graph	

		for TOP_K in 1 10
		do
			for PQ_SIZE in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 250 300 350 400 450 500 1000 1500 2000 2500
			do
				rm -rf build_template
				echo "DATASET:${DATASET_LOWER} TOP_K:${TOP_K} PQ_SIZE:${PQ_SIZE}" >> ./result/${PRINT_FILE}.txt
				./generate_template.sh && ./fill_parameters.sh ${PQ_SIZE} ${VEC_DIM} l2
			
				for ((i=1;i<=END;i++))
				do
					echo "COUNT:${i}" >> ./result/${PRINT_FILE}.txt
					./test_query.sh datasets/${DATASET}/${DATASET_LOWER}_query.fvecs datasets/${DATASET}/${DATASET_LOWER}_groundtruth.ivecs ${NUM_VEC} ${VEC_DIM} l2 ${TOP_K} ${SET_SINGLE_BATCH} >> ./result/${PRINT_FILE}.txt 2>&1
					echo >> ./result/${PRINT_FILE}.txt
				done
			done
		done
	done
done


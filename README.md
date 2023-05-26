# SONG with ADA-NNS

We have slightly modified the baseline code to run single batching. Please refer to original [SONG repository](https://github.com/SNU-ARC/song) for further details. For the moment, we only support l2 similarity metric.

## Prerequisites

Following are known versions to work. 

```shell
g++ 5.4.0, 9.3.0
CUDA 10.0, 10.1
```

## 1. Generate template to run code for specific dataset and priority queue size

VEC_DIM is vector dimension of the dataset. This value depends on the dataset. PQ_SIZE is priority queue size used in the approximate nearest neighbor search algorithm. Setting higher PQ_SIZE increases recall, however as mentioned in [README](https://github.com/SNU-ARC/song/blob/master/README.md), PQ_SIZE cannot be increased arbitrarily and it's maximum value depends on memory size.

```shell
Usage: ./generate_template.sh && ./fill_parameters.sh PQ_SIZE VEC_DIM l2
For example: ./generate_template.sh && ./fill_parameters.sh 50 128 l2
```

## 2. Build graph index to run approximate nearest neighbor search algorithm

BASE_VECTOR_PATH denotes path of vectors used to build graph. This yields bfsg.data and bfsg.graph, which would be used to run query. 

```shell
Usage: ./build_graph.sh BASE_VECTOR_PATH NUM_VEC VEC_DIM l2 
For example: ./build_graph.sh sift1m_base.fvecs 1000000 128 l2 
```

## 3. Run SONG GPU Searching Algorithm

We have added one more option as last argument of test_query.sh to run single batching.

```shell
Usage: ./test_query.sh QUERY_PATH GROUNDTRUTH_PATH NUM_VEC VEC_DIM l2 TOP_K IS_SINGLE_BATCH 
For example,
Run single batching: ./test_query.sh sift1m_query.fvecs sift1m_groundtruth.ivecs 1000000 128 l2 10 1
Run whole batching: ./test_query.sh sift1m_query.fvecs sift1m_groundtruth.ivecs 1000000 128 l2 10 0
```

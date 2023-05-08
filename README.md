
# SONG with ADA-NNS

This repository is SONG modified to run both single query and batch processing. 

Please refer to original [readme](https://github.com/SNU-ARC/song/blob/master/README.md).

### Prerequisites

Our prerequisites align with original SONG repository. Following are known versions to work. 

```shell
g++ 5.4.0, 9.3.0
CUDA 10.0, 10.1
```

## Usage

We provide script which builds graph, run queries and parses the result. All results can be found on folders result/song_single_batch, result/song_whole_batch. 

```shell
$ git clone https://github.com/SNU-ARC/song
$ cd song/
$ git checkout ADA-NNS
$ sudo ln -s YOUR_DATASETS_FOLDER datasets
$ ./script/run_whole.sh
```

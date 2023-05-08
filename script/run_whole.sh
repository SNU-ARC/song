#!/bin/sh

./script/make_graph.sh
./script/run.sh
python ./script/parse.py

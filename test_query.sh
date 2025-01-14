#!/bin/bash

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <query_data> <groundtruth_data> <built_graph_row> <built_graph_dimension> <l2/ip/cos> [display top k] <set_batch_size_1>" >&2
  echo "For example: $0 letter.scale.t 15000 26 cos" >&2
  echo "Use display top 5: $0 letter.scale.t 15000 26 cos 5" >&2
  exit 1
fi

search=100 #only used in CPU mode
display=100

if [ "$#" -ge 6 ]; then
  display=$6
fi
echo $display

$(dirname $0)/song test 0 $1 ${search} $3 $4 ${display} $5 $2 $7


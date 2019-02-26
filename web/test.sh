#!/usr/bin/env bash

set -e
set -x

DS_ID=$(curl -XPOST -F "file=@credit.csv" http://localhost:5000/dataset | jq -r .dataset_id)

curl -i http://localhost:5000/dataset/${DS_ID}

S_ID=$(curl -XPOST -d "dataset_id=$DS_ID&rows=500" http://localhost:5000/synthesis | jq -r .synthesis_id)

curl -i http://localhost:5000/synthesis/${S_ID}

#curl -i -XDELETE http://localhost:5000/dataset/${DS_ID}
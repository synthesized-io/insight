#!/usr/bin/env bash

set -e
set -x

BASE_URL=http://localhost:5000
#BASE_URL=https://webui.synthesized.io

DS_ID=$(curl -XPOST -F "file=@credit.csv" ${BASE_URL}/datasets | jq -r .dataset_id)

curl -i ${BASE_URL}/datasets/${DS_ID}

S_ID=$(curl -XPOST -d "dataset_id=$DS_ID&rows=500" ${BASE_URL}/syntheses | jq -r .synthesis_id)

curl -i ${BASE_URL}/syntheses/${S_ID}

curl -i -XDELETE ${BASE_URL}/datasets/${DS_ID}

#!/usr/bin/env bash

set -e
set -x

BASE_URL=http://localhost:5000
#BASE_URL=https://webui.synthesized.io

TOKEN=$(curl -f -XPOST -H "Content-Type: application/json" -d '{"username": "denis", "password": "123"}' ${BASE_URL}/auth | jq -r .access_token)
AUTH_HEADER="Authorization: JWT $TOKEN"

DS_ID=$(curl -f -XPOST -F "file=@credit.csv" -H "$AUTH_HEADER" ${BASE_URL}/datasets | jq -r .dataset_id)
if [[ -z ${DS_ID} ]];
then
    echo 'Could not upload dataset'
    exit 1
fi

curl -f -i -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}

curl -f -i -XPOST -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/model

S_ID=$(curl -f -XPOST -d "dataset_id=$DS_ID&rows=500" -H "$AUTH_HEADER" ${BASE_URL}/syntheses | jq -r .synthesis_id)

curl -f -i -H "$AUTH_HEADER" ${BASE_URL}/syntheses/${S_ID}

curl -f -i -XDELETE -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}

#!/usr/bin/env bash

set -e
set -x

BASE_URL=http://localhost:5000
#BASE_URL=https://webui.synthesized.io

TOKENS_JSON=$(curl -f -XPOST -H "Content-Type: application/json" -d '{"username": "denis", "password": "123"}' ${BASE_URL}/login)
ACCESS_TOKEN=$(echo ${TOKENS_JSON} | jq -r .access_token)
REFRESH_TOKEN=$(echo ${TOKENS_JSON} | jq -r .refresh_token)
AUTH_HEADER="Authorization: Bearer $ACCESS_TOKEN"
REFRESH_HEADER="Authorization: Bearer $REFRESH_TOKEN"

curl -f -i -XPOST -H "$REFRESH_HEADER" ${BASE_URL}/refresh

DS_ID=$(curl -f -XPOST -F "file=@credit.csv" -H "$AUTH_HEADER" ${BASE_URL}/datasets | jq -r .dataset_id)
if [[ -z ${DS_ID} ]];
then
    echo 'Could not upload dataset'
    exit 1
fi

curl -f -i -XPOST -d 'title=title&description=description' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/updateinfo

curl -f -i -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}

curl -f -i -XPOST -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/model

while true; do
    STATUS=$(curl -f -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/model-training | jq -r .status)
    if [[ "$STATUS" != "training" ]]; then
        break
    fi
    sleep 5
done


S_ID=$(curl -f -XPOST -d "dataset_id=$DS_ID&rows=500" -H "$AUTH_HEADER" ${BASE_URL}/syntheses | jq -r .synthesis_id)

curl -f -i -H "$AUTH_HEADER" ${BASE_URL}/syntheses/${S_ID}

curl -f -i -XDELETE -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}

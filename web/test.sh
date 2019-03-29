#!/usr/bin/env bash

set -e
set -x

BASE_URL=http://localhost:5000/api
#BASE_URL=https://webui.synthesized.io/api

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
    STATUS=$(curl -f -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/model | jq -r .status)
    if [[ "$STATUS" != "training" ]]; then
        break
    fi
    sleep 5
done

curl -f -i -XPOST -d 'rows=10000' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/synthesis

curl -f -i -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/synthesis

CORR_ID=$(curl -f -XPOST -d 'type=CORRELATION' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items | jq -r .id)
MOD_ID=$(curl -f -XPOST -d 'type=MODELLING' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items | jq -r .id)

curl -f -XPOST -d 'new_order=0' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items/${MOD_ID}/move
curl -f -XPOST -d 'new_order=1' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items/${MOD_ID}/move

curl -f -i -XPOST -d '{"settings": {"columns": ["age", "MonthlyIncome"]}, "max_sample_size": 10}' -H 'Content-Type: application/json' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items/${CORR_ID}/updatesettings
curl -f -i -XPOST -d '{"settings": {"response_variable": "SeriousDlqin2yrs", "explanatory_variables": ["NumberOfTimes90DaysLate", "age", "effort", "MonthlyIncome"], "model": "LogisticRegression"}}' -H 'Content-Type: application/json' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items/${MOD_ID}/updatesettings
curl -f -i -XPOST -d '{"settings": {"response_variable": "MonthlyIncome", "explanatory_variables": ["NumberOfTimes90DaysLate", "age", "effort", "SeriousDlqin2yrs"], "model": "GradientBoostingRegressor"}}' -H 'Content-Type: application/json' -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items/${MOD_ID}/updatesettings

curl -f -i -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report

curl -f -XDELETE -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}/report-items/${MOD_ID}

curl -f -i -XDELETE -H "$AUTH_HEADER" ${BASE_URL}/datasets/${DS_ID}

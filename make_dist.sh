#!/usr/bin/env bash
set -ex

VERSION=$(cat synthesized/version.py | cut -f2 -d'=' | cut -c 2- | tr -d "'")
TIMETAG=$(date +"%Y%m%d")

DIST_NAME=dist/synthesized_distribution_${VERSION}_${TIMETAG}
mkdir -p ${DIST_NAME}

make build
cp dist/*.whl ${DIST_NAME}/
pip download -r requirements.txt -d ${DIST_NAME}/deps
git rev-parse HEAD > ${DIST_NAME}/revision.txt

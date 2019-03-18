#!/usr/bin/env bash

set -e
set -x

for ARCHIVE in dist/*.whl; do
    for F in $(zipinfo -1 ${ARCHIVE} | grep "\.py$" | grep -v '__init__\.py'); do
        zip -d ${ARCHIVE} "${F}"
    done
done

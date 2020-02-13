#!/usr/bin/env bash

set -exo pipefail

SCRIPT="$1"

if [[ -z "${SCRIPT}" ]]; then
    >&2 echo 'No script given'
    exit 1
fi

ROOT_DIR="$(cd "$(dirname $(dirname "$0"))" ; pwd -P )"
cd "${ROOT_DIR}"

ray submit tune/tune-default.yaml ${SCRIPT} --start

#!/usr/bin/env bash

set -e
set -x

DIR=$(cd $(dirname "$0") && pwd)
cd $(dirname ${DIR})

FLASK_ENV=development FLASK_APP=web.app:app python3 -m flask run

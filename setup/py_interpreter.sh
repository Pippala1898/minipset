#!/usr/bin/env bash

readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}"c)")

/opt/conda/bin/python "$@"

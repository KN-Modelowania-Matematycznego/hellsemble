#!/bin/bash
set -e

export PATHS_TO_CHECK="hellsemble/ test/"

echo "Running isort"
isort --profile=black --line-length=79 $PATHS_TO_CHECK

echo "Running black"
black --line-length=79 $PATHS_TO_CHECK

echo "Running flake8"
flake8 --ignore=W605,W503 $PATHS_TO_CHECK
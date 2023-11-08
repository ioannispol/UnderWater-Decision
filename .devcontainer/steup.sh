#!/bin/bash

git config --global --add safe.directory /workspaces/UnderWater-Decision

pip install -e .[dev]
pip install pytest-cov
pre-commit install

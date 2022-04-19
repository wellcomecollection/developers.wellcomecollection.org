#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o verbose

NODE_IMAGE="760097843905.dkr.ecr.eu-west-1.amazonaws.com/node:16-alpine"
ROOT=$(git rev-parse --show-toplevel)

docker run --rm --tty \
  --volume "$ROOT:$ROOT" \
  --workdir "$ROOT" \
  --env SSH=true \
  "$NODE_IMAGE" yarn deploy

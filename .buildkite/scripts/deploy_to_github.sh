#!/usr/bin/env bash
# This deploys a collection of built HTML files to GitHub.
#
# Because this involves Git, we can't use the Alpine Node image
# we do to build the HTML -- it doesn't have Git installed.
# We have to build a custom image.

set -o errexit
set -o nounset
set -o verbose

ROOT=$(git rev-parse --show-toplevel)
DEPLOY_IMAGE="node_with_git"

docker build \
  --tag "$DEPLOY_IMAGE" \
  --file "$ROOT/.buildkite/scripts/deploy.Dockerfile" .

docker run --rm --tty \
  --volume "$ROOT:$ROOT" \
  --volume ~/.ssh:/root/.ssh \
  --volume ~/.gitconfig:/root/.gitconfig \
  --workdir "$ROOT" \
  --env SSH=true \
  "$DEPLOY_IMAGE" yarn deploy

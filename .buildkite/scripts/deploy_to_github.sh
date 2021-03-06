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

# If we don't set up our Git config, the "deploy" command further
# down will fail and prompt us to run these commands.
if [[ "$CI" == "true" ]]
then
  git config --global user.email "wellcomedigitalplatform@wellcome.ac.uk"
  git config --global user.name "Buildkite on behalf of Wellcome Collection"
  # Stop git complaining about the repo being owned by a different account
  git config --global --add safe.directory $ROOT
fi

docker build \
  --tag "$DEPLOY_IMAGE" \
  --file "$ROOT/.buildkite/scripts/deploy.Dockerfile" .

docker run --rm --tty \
  --volume "$ROOT:$ROOT" \
  --workdir "$ROOT" \
  "$DEPLOY_IMAGE" yarn

# Buildkite has already added the weco-bot SSH key to its ssh-agent,
# so we use the agent from the host as per:
# https://www.jamesridgway.co.uk/sharing-an-ssh-agent-between-a-host-machine-and-a-docker-container/
# We also use the .ssh config dir from the host so we get the github.com fingerprint in our known_hosts
docker run --rm --tty \
  --volume "$ROOT:$ROOT" \
  --volume ~/.gitconfig:/root/.gitconfig \
  --volume ~/.ssh:/root/.ssh \
  --volume $SSH_AUTH_SOCK:/ssh-agent \
  --env SSH_AUTH_SOCK=/ssh-agent \
  --workdir "$ROOT" \
  --env USE_SSH=true \
  "$DEPLOY_IMAGE" yarn deploy

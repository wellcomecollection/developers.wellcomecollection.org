#!/usr/bin/env bash
# This script converts the example notebooks in the repo to markdown, which is
# then used to build the HTML site.

set -o errexit
set -o nounset

root=$(git rev-parse --show-toplevel)

# delete everything in the docs/exaples folder
rm -rf $root/docs/examples/*.md

# strip output from notebooks
docker run --rm --tty \
  --volume "$root:$root" \
  --workdir "$root" \
  --user root \
    jupyter/scipy-notebook \
    jupyter nbconvert --clear-output --inplace $root/notebooks/*.ipynb

# convert notebooks to markdown
docker run --rm --tty \
  --volume "$root:$root" \
  --workdir "$root" \
  --user root \
    jupyter/scipy-notebook \
    jupyter nbconvert \
    --to markdown \
    --template .buildkite/scripts/mdoutput \
    --output-dir "$root/docs/examples" \
    $root/notebooks/*.ipynb

# create an index.md file for the examples folder with a table of contents
cat << EOF > "$root/docs/examples/index.md"
---
title: Examples
---

# Examples
EOF

# cat the introduction.md file into index.md
cat "$root/notebooks/introduction.md" >> "$root/docs/examples/index.md"

# write a table of contents in index.md
cat << EOF >> "$root/docs/examples/index.md"

## Table of contents

EOF

# loop through the files in the examples folder
for file in $root/docs/examples/*.md; do
  if [[ "$file" != "$root/docs/examples/index.md" ]]; then
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    # get the index of the file
    index="${filename%%-*}"
    # remove numbers from the start of the filename
    filename="${filename#[0-9]*-}"
    # replace hyphens with spaces
    title="${filename//-/ }"
    # capitalise first letter of the title
    title="$(tr '[:lower:]' '[:upper:]' <<< ${title:0:1})${title:1}"
    echo "$index. [$title](/docs/examples/$filename)" >> "$root/docs/examples/index.md"
  fi
done


# add a link to github and colab for each notebook
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

for file in $root/notebooks/*.ipynb; do
  filename=$(basename -- "$file")

  github_path="wellcomecollection/developers.wellcomecollection.org/tree/$GIT_BRANCH/notebooks/$filename"
  github_url="https://github.com/$github_path"
  colab_url="https://colab.research.google.com/github/$github_path"

  line="\n[View on GitHub]($github_url) | [Run in Google Colab]($colab_url)"
  path="$root/docs/examples/${filename%.*}.md"
  # insert the line at the second line of the file
  awk -v line="$line" 'NR==2{print line}1' "$path" > tmp && mv -f tmp "$path"
done

# commit any changes back to the branch
if [[ `git status --porcelain` ]]; then
  git config user.name "GitHub on behalf of Wellcome Collection"
  git config user.email "wellcomedigitalplatform@wellcome.ac.uk"

  git add --verbose --update
  git commit -m "Convert notebooks"

  git push
  exit 1;
else
  echo "No changes from notebook conversion"
  exit 0;
fi

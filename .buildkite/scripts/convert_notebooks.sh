#!/usr/bin/env bash
# This script converts the example notebooks in the repo to markdown, which is
# then used to build the HTML site.

set -o errexit
set -o nounset
set -o verbose

root=$(git rev-parse --show-toplevel)

# delete everything in the docs/exaples folder
rm -rf "$root/docs/examples"/*

docker run --rm --tty \
  --volume "$root:$root" \
  --workdir "$root" \
    jupyter/scipy-notebook \
    jupyter nbconvert \
    --to markdown \
    --output-dir "$root/docs/examples" \
    "$root/notebooks"/**.ipynb

# create an index.md file for the examples folder with a table of contents

cat << EOF > "$root/docs/examples/index.md"
---
title: Examples
---

# Examples

EOF

# write a table of contents in index.md
for file in "$root/docs/examples"/*.md; do
  if [[ "$file" != "$root/docs/examples/index.md" ]]; then
    filename=$(basename -- "$file")
    filename="${filename%.*}"
    # remove numbers from the start of the filename
    filename="${filename#[0-9]*-}"
    # replace hyphens with spaces
    title="${filename//-/ }"
    # capitalise first letter of the title
    title="$(tr '[:lower:]' '[:upper:]' <<< ${title:0:1})${title:1}"
    echo "- [$title](/docs/examples/$filename)" >> "$root/docs/examples/index.md"
  fi
done


# add a link to github and colab for each notebook
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

for file in "$root/notebooks"/*.ipynb; do
  filename=$(basename -- "$file")

  github_path="wellcomecollection/developers.wellcomecollection.org/tree/$GIT_BRANCH/notebooks/$filename"
  github_url="https://github.com/$github_path"
  colab_url="https://colab.research.google.com/github/$github_path"

  line="\n[View on GitHub]($github_url) | [Run in Google Colab]($colab_url)"
  path="$root/docs/examples/${filename%.*}.md"
  # insert the line at the second line of the file
  awk -v line="$line" 'NR==2{print line}1' "$path" > tmp && mv tmp "$path"
done

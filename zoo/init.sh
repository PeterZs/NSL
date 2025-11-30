#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 遍历脚本所在目录下的所有子文件夹
for dir in "$SCRIPT_DIR"/*/; do
  dir_name=$(basename "$dir")

  # exlude dirs named with ckpts
  if [ "$dir_name" != "ckpts" ] && [ -d "$dir" ]; then
    touch "${dir}__init__.py"
    echo "Created __init__.py in $dir"
  fi
done
#!/bin/bash

DEFAULT_INPUT_DIR="resources/data/openml"
DEFAULT_OTUPUT_DIR="resources/data/openml-restructured"
DEFAULT_COPY=true

INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OTUPUT_DIR}"
COPY="${3:-$DEFAULT_COPY}"

for split in train test; do
    for filepath in $INPUT_DIR/$split/*.csv; do
        filename=$(basename "$filepath")            
        dataname="${filename%.csv}"            

        mkdir -p "$OUTPUT_DIR/$dataname/train"  
        mkdir -p "$OUTPUT_DIR/$dataname/test"            

        echo "$filepath -> $target_path"
        if [[ "$COPY" == true ]]; then
            cp "$filepath" "$OUTPUT_DIR/$dataname/$split/$split.csv"
        else
            mv "$filepath" "$OUTPUT_DIR/$dataname/$split/$split.csv"
        fi
    done
done

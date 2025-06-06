#!/bin/bash

# Parameters (edit these values as needed)
ORIGINAL_NUM=15
COPY_NUM=187

# Directory containing the JSON files (use "." for current directory)
DIR="."

# Loop through all JSON files starting with "N${ORIGINAL_NUM}_"
for file in "$DIR"/N${ORIGINAL_NUM}_*.json; do
  # Skip if no files match
  [ -e "$file" ] || continue

  # Create new filename by replacing OLD_NUM with NEW_NUM
  new_file="${file/N${ORIGINAL_NUM}_/N${COPY_NUM}_}"

  # Replace "number": ORIGINAL_NUM with "number": COPY_NUM and save to new file
  sed "s/\"number\":[[:space:]]*$ORIGINAL_NUM/\"number\": $COPY_NUM/g" "$file" > "$new_file"

  echo "Created $new_file"
done

#!/bin/bash

# Loop over all files with the extension .img.nii.gz in the current directory
for file in *.img.nii.gz
do
  # Use basename to extract the filename before .img.nii.gz
  folder_name=$(basename "$file" .img.nii.gz)

  # Create a folder with the extracted number if it doesn't already exist
  mkdir -p "$folder_name"

  # Output the result
  echo "Created folder $folder_name and moved $file into it."
done


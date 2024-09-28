#!/bin/bash

# Loop through all .img.nrrd files in the current directory
for file in *.img.nii.gz; do
  # Extract the base name of the file
  base=$(basename "$file" .img.nii.gz)

  # Print the base name to check it
  echo "Processing file: $file with base name: $base"

  # Here you can add your command to process the file
  # Example:
  TotalSegmentator -i "$file" -o "$base" --roi_subset vertebrae_C3 vertebrae_C5
done


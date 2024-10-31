#!/bin/bash

# Loop through all .img.nrrd files in the current directory
for file in *.img.nii.gz; do
  # Extract the base name of the file
  base=$(basename "$file" .img.nii.gz)

  # Print the base name to check it
  echo "Processing file: $file with base name: $base"

  # Segment common carotid artery and vertebrae
  TotalSegmentator -i "$file" -o "$base" --roi_subset common_carotid_artery_right common_carotid_artery_left vertebrae_C3 vertebrae_C5
done


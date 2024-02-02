import SimpleITK as sitk
import numpy as np

def load_seed(seed_image_path):

    seed_image = sitk.ReadImage(seed_image_path)
    seed_indices = sitk.GetArrayFromImage(seed_image)
    seeds = list(zip(*seed_indices.nonzero()))
    seeds = [tuple(int(x) for x in seed) for seed in seeds]
    seeds = [(seed[2], seed[1], seed[0]) for seed in seeds]
    return seeds

# load images
cta_image_path = '62.img.nrrd'
cta_image = sitk.ReadImage(cta_image_path)
cta_image_array = sitk.GetArrayFromImage(cta_image)

#load carotid artery segmentations
seed_image_path_left = 'common_carotid_artery_left.nii.gz'
seed_image_path_right = 'common_carotid_artery_right.nii.gz'
seeds = load_seed(seed_image_path_left)
seeds += load_seed(seed_image_path_right)

# Set threshold
lower_threshold = 200
upper_threshold = 300
output_image = sitk.ConnectedThreshold(cta_image, seedList=seeds,
                                      lower=lower_threshold, upper=upper_threshold)
# output_image = sitk.BinaryThreshold(cta_image, lower_threshold, upper_threshold, 1, 0)


# Remove unnaccessary part
output_image_array = sitk.GetArrayFromImage(output_image)
output_image_array[:40] = np.zeros_like(output_image_array[:40])

# Add header and make output file
output_image_modified = sitk.GetImageFromArray(output_image_array)
output_image_modified.CopyInformation(output_image)
output_image_modified.SetSpacing(output_image.GetSpacing())
output_image_modified.SetOrigin(output_image.GetOrigin())
output_image_modified.SetDirection(output_image.GetDirection())
output_image_path = 'output_image.nrrd'
sitk.WriteImage(output_image_modified, output_image_path)

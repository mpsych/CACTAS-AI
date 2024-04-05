import SimpleITK as sitk
import numpy as np

def segment_carotid_artery_levelset(preprocessed_image_sitk, seed_points):
    # Convert the image to a float type needed for the level set filters.
    image_float = sitk.Cast(preprocessed_image_sitk, sitk.sitkFloat64)

    # Initialize the level set image with zeros.
    init_ls = sitk.Image(image_float.GetSize(), sitk.sitkFloat64)
    init_ls.CopyInformation(image_float)

    # Set seed points with high positive values to initialize the segmentation.
    for seed in seed_points:
        init_ls[seed] = 1

    # Use the Fast Marching Method to generate the initial level set.
    fast_marching = sitk.FastMarchingImageFilter()
    fast_marching.SetTrialPoints(seed_points)
    fast_marching.SetStoppingValue(1)
    initial_level_set = fast_marching.Execute(init_ls)

    # Configure the Geodesic Active Contour Level Set filter.
    geodesicActiveContourLS = sitk.GeodesicActiveContourLevelSetImageFilter()
    geodesicActiveContourLS.SetPropagationScaling(1.0)
    geodesicActiveContourLS.SetCurvatureScaling(1.0)
    geodesicActiveContourLS.SetAdvectionScaling(1.0)
    geodesicActiveContourLS.SetMaximumRMSError(0.02)
    geodesicActiveContourLS.SetNumberOfIterations(200)

    # Apply the filter with the initial level set image.
    segmented_image = geodesicActiveContourLS.Execute(initial_level_set, image_float)

    return segmented_image


def load_seed(seed_image_path, cta_image):
    seed_image = sitk.ReadImage(seed_image_path)
    seed_indices = sitk.GetArrayFromImage(seed_image)
    seeds = list(zip(*seed_indices.nonzero()))
    seeds = [tuple(int(x) for x in seed) for seed in seeds]
    seeds = [(seed[2], seed[1], seed[0]) for seed in seeds]

    intensity_values = []
    for seed in seeds:
        # Extract a small region around the seed point
        region_radius = [1, 1, 1]  # Define the size of the region around the seed point
        region = sitk.RegionOfInterest(cta_image, region_radius, seed)
        region_array = sitk.GetArrayFromImage(region)

        # Collect all intensity values of the region
        intensity_values.extend(region_array.flatten())

    return seeds, intensity_values


# Now integrate the preprocess step in the main workflow
if __name__ == "__main__":
    cta_image_path = '11/11.b.img.nrrd'
    cta_image = sitk.ReadImage(cta_image_path)
    cta_image_array = sitk.GetArrayFromImage(cta_image)

    # load carotid artery segmentations
    seed_image_path_left = '11/common_carotid_artery_left.nii.gz'
    seed_image_path_right = '11/common_carotid_artery_right.nii.gz'
    seeds_left, intensities_left = load_seed(seed_image_path_left, cta_image)
    seeds_right, intensities_right = load_seed(seed_image_path_right, cta_image)

    # Combine seeds and intensity values from both sides
    seeds = seeds_left + seeds_right
    intensity_values = intensities_left + intensities_right

    intensity_values_array = np.array(intensity_values)
    intensity_values_array = intensity_values_array[intensity_values_array >= 0]
    sorted_intensity_values = np.sort(intensity_values_array)

    median_intensity = np.median(sorted_intensity_values)
    mean_intensity_1 = np.mean(sorted_intensity_values)
    mean_intensity_2 = np.mean(intensity_values_array)
    # Determine global minimum and maximum intensity values
    min_intensity = min(sorted_intensity_values)
    max_intensity = max(sorted_intensity_values)

    # Use global min and max as the thresholds
    lower_threshold = float(median_intensity)
    upper_threshold = lower_threshold + 100

    # Use adjusted thresholds in the ConnectedThreshold function
    output_image = sitk.ConnectedThreshold(cta_image, seedList=seeds,
                                           lower=lower_threshold, upper=upper_threshold)

    seed_indices = sitk.GetArrayFromImage(output_image)
    all_seeds = list(zip(*seed_indices.nonzero()))
    all_seeds = [tuple(int(x) for x in seed) for seed in all_seeds]
    all_seeds = [(seed[2], seed[1], seed[0]) for seed in all_seeds]


    segmented_image = segment_carotid_artery_levelset(cta_image, all_seeds)
    segmented_image_array = sitk.GetArrayFromImage(segmented_image)

    # Save the segmented image
    output_image_path = 'segmented_levelset_11_6.nrrd'
    sitk.WriteImage(segmented_image, output_image_path)

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

#preprocess_image(file_sitk, lower_bound=-240.0, upper_bound=160.0, smoothing=False, contrast_enhancement=False):
def preprocess_image(file_sitk, window_width, window_level, smoothing=False, contrast_enhancement=False):
    image_data = sitk.GetArrayFromImage(file_sitk)

    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)

    image_data_windowed = np.clip(image_data, lower_bound, upper_bound)

    if smoothing:
        image_data_smoothed = sitk.CurvatureFlow(image1=file_sitk,
                                                 timeStep=0.125,
                                                 numberOfIterations=5)
        image_data_windowed = sitk.GetArrayFromImage(image_data_smoothed)


    image_data_pre = (image_data_windowed - np.min(image_data_windowed)) / (
                np.max(image_data_windowed) - np.min(image_data_windowed)) * 255.0
    image_data_pre = np.uint8(image_data_pre)

    if contrast_enhancement:
        image_sitk_pre = sitk.GetImageFromArray(image_data_pre)
        image_data_pre = sitk.GetArrayFromImage(sitk.AdaptiveHistogramEqualization(image_sitk_pre))

    preprocessed_image_sitk = sitk.GetImageFromArray(image_data_pre)
    preprocessed_image_sitk.CopyInformation(file_sitk)

    return preprocessed_image_sitk



def level_set(preprocessed_image, seed_points):
    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussian(preprocessed_image, sigma=1.0)
    gradient_magnitude_array = sitk.GetArrayFromImage(gradient_magnitude)

    # Create an image to use as the initial level set function, which is the same size as the input image.
    seed_image = sitk.Image(preprocessed_image.GetSize(), sitk.sitkFloat32)
    seed_image.CopyInformation(preprocessed_image)

    # Set all pixels in the seed image to background value (e.g., 1)
    # seed_image += 1

    # Set seed points to foreground value (e.g., 0)
    for point in seed_points:
        seed_image[point] = 70

    # Set up the level set filter with selected parameters
    geodesicActiveContourLS = sitk.GeodesicActiveContourLevelSetImageFilter()
    geodesicActiveContourLS.SetPropagationScaling(20.0)  # Increased to expand more
    geodesicActiveContourLS.SetCurvatureScaling(0.1)  # A bit higher to smooth over details
    geodesicActiveContourLS.SetAdvectionScaling(0.5)  # Reduced to decrease edge attraction
    geodesicActiveContourLS.SetMaximumRMSError(0.02)
    geodesicActiveContourLS.SetNumberOfIterations(1000)

    # Run the level set algorithm
    ls_result = geodesicActiveContourLS.Execute(seed_image, gradient_magnitude)
    ls_result_array = sitk.GetArrayFromImage(ls_result)
    return ls_result


def segment_carotid_artery_levelset(preprocessed_image_sitk, seed_points):
    # Assuming 'preprocessed_image_sitk' is your 3D image already loaded into SimpleITK.
    image_float = sitk.Cast(preprocessed_image_sitk, sitk.sitkFloat64)

    # Apply a gradient magnitude filter to enhance edges.
    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradient_magnitude.SetSigma(1.0)  # Adjust sigma based on your image's characteristics.
    feature_image = gradient_magnitude.Execute(image_float)
    feature_image = sitk.Cast(feature_image, sitk.sitkFloat64)
    feature_image_array = sitk.GetArrayFromImage(feature_image)

    # Initialize the level set image with zeros and copy metadata from the feature image.
    init_ls = sitk.Image(feature_image.GetSize(), sitk.sitkFloat64)
    init_ls.CopyInformation(feature_image)
    # feature_image_2 = -image_float

    # Apply seeds with high positive values to initialize the segmentation.
    # for seed in seed_points:
    #     init_ls[seed] = 1.0  # Ensure seed points are in the correct format and within the image bounds.

    # init_ls_array = sitk.GetArrayFromImage(init_ls)
    # feature_image_2_array = sitk.GetArrayFromImage(feature_image_2)

    # Use the Fast Marching Method to generate the initial level set.
    fast_marching = sitk.FastMarchingImageFilter()
    fast_marching.SetTrialPoints(seed_points)
    fast_marching.SetStoppingValue(-180.0)
    initial_level_set = fast_marching.Execute(-image_float)
    initial_level_set_array = sitk.GetArrayFromImage(initial_level_set)

    # Configure the Geodesic Active Contour Level Set filter.
    geodesicActiveContourLS = sitk.GeodesicActiveContourLevelSetImageFilter()
    geodesicActiveContourLS.SetPropagationScaling(10.0)
    geodesicActiveContourLS.SetCurvatureScaling(0.01)
    geodesicActiveContourLS.SetAdvectionScaling(1.0)
    geodesicActiveContourLS.SetMaximumRMSError(0.05)
    geodesicActiveContourLS.SetNumberOfIterations(1000)

    segmented_image = geodesicActiveContourLS.Execute(initial_level_set, image_float)
    segmented_image_array = sitk.GetArrayFromImage(segmented_image)
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
    cta_image_path = '55/55.b.image.nii.gz'
    cta_image = sitk.ReadImage(cta_image_path)
    cta_image_array = sitk.GetArrayFromImage(cta_image)

    # load carotid artery segmentations
    seed_image_path_left = '55/common_carotid_artery_left.nii.gz'
    seed_image_path_right = '55/common_carotid_artery_right.nii.gz'
    seeds_left, intensities_left = load_seed(seed_image_path_left, cta_image)
    seeds_right, intensities_right = load_seed(seed_image_path_right, cta_image)


    # Combine seeds and intensity values from both sides
    seeds = seeds_left + seeds_right
    intensity_values = intensities_left + intensities_right

    intensity_values_array = np.array(intensity_values)
    intensity_values_array = intensity_values_array[intensity_values_array >= 0]
    sorted_intensity_values = np.sort(intensity_values_array)

    #median_intensity = np.median(sorted_intensity_values)
    mean_intensity_1 = np.mean(sorted_intensity_values)
    #mean_intensity_2 = np.mean(intensity_values_array)

    # Determine global minimum and maximum intensity values
    min_intensity = min(sorted_intensity_values)
    max_intensity = max(sorted_intensity_values)






    intraluminal_HU = np.mean(sorted_intensity_values)
    window_width = intraluminal_HU * 2.07
    window_level = intraluminal_HU * 0.72


    preprocessed_image_sitk = preprocess_image(cta_image, window_width=window_width,
                                               window_level=window_level,
                                               smoothing=False, contrast_enhancement=True)
    pre_image_array = sitk.GetArrayFromImage(preprocessed_image_sitk)

    #plt.imshow(pre_image_array[100,:,:])
    #plt.show()



    output_image_path = 'preprocessed_55.img.nrrd'
    sitk.WriteImage(preprocessed_image_sitk, output_image_path)

    seeds_left, intensities_left = load_seed(seed_image_path_left, preprocessed_image_sitk)
    seeds_right, intensities_right = load_seed(seed_image_path_right, preprocessed_image_sitk)






    # Combine seeds and intensity values from both sides
    seeds = seeds_left + seeds_right
    intensity_values = intensities_left + intensities_right

    intensity_values_array = np.array(intensity_values)
    intensity_values_array = intensity_values_array[intensity_values_array >= 0]
    sorted_intensity_values = np.sort(intensity_values_array)

    # median_intensity = np.median(sorted_intensity_values)
    mean_intensity_1 = np.mean(sorted_intensity_values)
    # mean_intensity_2 = np.mean(intensity_values_array)








    # Use global min and max as the thresholds
    lower_threshold = float(mean_intensity_1)
    upper_threshold = lower_threshold + 30


    # Use adjusted thresholds in the ConnectedThreshold function
    output_image = sitk.ConnectedThreshold(preprocessed_image_sitk, seedList=seeds,
                                           lower=lower_threshold, upper=upper_threshold)


    seed_indices = sitk.GetArrayFromImage(output_image)
    all_seeds = list(zip(*seed_indices.nonzero()))
    all_seeds = [tuple(int(x) for x in seed) for seed in all_seeds]
    all_seeds = [(seed[2], seed[1], seed[0]) for seed in all_seeds]


    segmented_image = segment_carotid_artery_levelset(preprocessed_image_sitk, seeds)
    # segmented_image = level_set(preprocessed_image_sitk, seeds)
    segmented_image_array = sitk.GetArrayFromImage(segmented_image)


    # Save the segmented image
    output_image_path = 'norm_thresh_levelset_11_9.nrrd'
    sitk.WriteImage(segmented_image, output_image_path)
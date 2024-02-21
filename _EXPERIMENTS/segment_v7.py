import SimpleITK as sitk
import numpy as np

def preprocess_image(file_sitk, lower_bound=-240.0, upper_bound=160.0): # 155.0 600.0
    """
    Preprocess the image by applying windowing and normalization.
    """
    image_data = sitk.GetArrayFromImage(file_sitk)
    # Apply windowing
    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
    # Normalize to [0, 255]
    image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
    image_data_pre = np.uint8(image_data_pre)
    # Convert back to SimpleITK image
    preprocessed_image_sitk = sitk.GetImageFromArray(image_data_pre)
    preprocessed_image_sitk.CopyInformation(file_sitk)

    return preprocessed_image_sitk

def segment_carotid_artery(cta_image, seeds, lower_threshold, upper_threshold):
    """
    Segments the carotid artery by applying region growing and refining with the level set method.
    """
    # Region growing based on intensity and seed points
    initial_segmentation = sitk.ConnectedThreshold(image1=cta_image, seedList=seeds,
                                                   lower=lower_threshold, upper=upper_threshold,
                                                   replaceValue=1)

    # Convert the region growing result into a signed distance map for the level set method
    initial_level_set = sitk.SignedMaurerDistanceMap(initial_segmentation, insideIsPositive=False, useImageSpacing=True)

    # Calculate the gradient magnitude of the CTA image
    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussian(cta_image, sigma=1.0)

    # Refine segmentation with the level set method
    refined_segmentation = sitk.GeodesicActiveContourLevelSet(initial_level_set, gradient_magnitude,
                                                              propagationScaling=1.0, curvatureScaling=2.0,
                                                              advectionScaling=1.0, maximumRMSError=0.02,
                                                              numberOfIterations=500)

    # Convert the refined segmentation to a binary image
    final_segmentation = sitk.BinaryThreshold(refined_segmentation, lowerThreshold=0, upperThreshold=0.5, insideValue=1,
                                              outsideValue=0)

    return final_segmentation


def create_initial_segmentation(image, seed_points):
    # Create a blank image with the same dimensions as the input image
    initial_segmentation = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    initial_segmentation.CopyInformation(image)

    # Optionally, set seed points in initial_segmentation if needed
    # This step depends on how you intend to use the seed points

    return initial_segmentation


def segment_carotid_artery_levelset_only(cta_image, seed_points, sigma=1.0):
    """
    Segments the carotid artery using the level set method without initial thresholding.
    """
    # Assuming initial_segmentation_seed is a broad segmentation or an anatomical model
    # as the starting point for the level set method.
    initial_segmentation_seed = create_initial_segmentation(cta_image, seed_points)
    # Convert the initial segmentation seed into a signed distance map for the level set method
    initial_level_set = sitk.SignedMaurerDistanceMap(initial_segmentation_seed, insideIsPositive=False, useImageSpacing=True)
    initial_image_array = sitk.GetArrayFromImage(initial_level_set)
    # Calculate the gradient magnitude of the CTA image
    gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussian(cta_image, sigma=sigma)

    # Refine segmentation with the level set method
    refined_segmentation = sitk.GeodesicActiveContourLevelSet(initial_level_set, gradient_magnitude,
                                                              propagationScaling=1.0, curvatureScaling=2.0,
                                                              advectionScaling=1.0, maximumRMSError=0.02,
                                                              numberOfIterations=500)

    # Convert the refined segmentation to a binary image
    final_segmentation = sitk.BinaryThreshold(refined_segmentation, lowerThreshold=0, upperThreshold=0.5, insideValue=1,
                                              outsideValue=0)

    return final_segmentation

def load_seed(seed_image_path, cta_image):
    seed_image = sitk.ReadImage(seed_image_path)
    seed_indices = sitk.GetArrayFromImage(seed_image)
    seeds = list(zip(*seed_indices.nonzero()))
    seeds = [tuple(int(x) for x in seed) for seed in seeds]
    seeds = [(seed[2], seed[1], seed[0]) for seed in seeds]
    return seeds

# Now integrate the preprocess step in the main workflow
if __name__ == "__main__":
    demo_file_nii = "11/11.b.img.nrrd"
    file_sitk = sitk.ReadImage(demo_file_nii)

    # Preprocess the image
    preprocessed_image_sitk = preprocess_image(file_sitk)
    pre_image_array = sitk.GetArrayFromImage(preprocessed_image_sitk)

    # Continue with the segmentation process...
    # For example, load seeds and apply segmentation on the preprocessed image
    seed_image_path_left = '11/common_carotid_artery_left.nii.gz'
    seed_image_path_right = '11/common_carotid_artery_right.nii.gz'
    seeds_left = load_seed(seed_image_path_left, preprocessed_image_sitk)
    seeds_right = load_seed(seed_image_path_right, preprocessed_image_sitk)
    seed_points = seeds_left + seeds_right

    lower_threshold, upper_threshold = 120,200

    # Assuming the segment_carotid_artery function is defined and integrates the seeds and preprocessed image
    segmented_image = segment_carotid_artery_levelset_only(preprocessed_image_sitk, seed_points)

    # Save the segmented image
    output_image_path = 'segmented_carotid_artery11_2.nrrd'
    sitk.WriteImage(segmented_image, output_image_path)

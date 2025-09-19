"""Calculate average MS-SSIM scores for dataset quality assessment.

This script analyzes the MS-SSIM (Multi-Scale Structural Similarity) scores of images
in the RawNIND dataset to evaluate image quality distribution. MS-SSIM is a perceptual
metric that measures image similarity based on luminance, contrast, and structure at
multiple scales, with values ranging from 0 (completely different) to 1 (identical).

The script performs the following operations:
1. Loads the RawNIND dataset descriptor containing image metadata and MS-SSIM scores
2. Loads the test_reserve configuration to identify test set images
3. Filters images by:
   - Bayer pattern images only (is_bayer=True)
   - Images that have RGB MS-SSIM scores
   - Images that belong to the test reserve set
4. Groups image crops by MS-SSIM threshold values
5. Calculates the average MS-SSIM score for each threshold group
6. Reports statistics about crop counts and average scores by threshold

This analysis helps understand the quality distribution in the test dataset and can
be used to:
- Ensure balanced representation of image qualities in test sets
- Select appropriate threshold values for model evaluation
- Compare quality distributions between training and test sets
- Guide dataset filtering for targeted model training

The results section contains historical analysis results for reference,
showing threshold-based statistics for both "less than or equal to" (le) and 
"greater than or equal to" (ge) filtering approaches.
"""

import statistics
import sys

import yaml

# Add parent directory to path for imports
sys.path.append('..')
from .libs import rawproc


def load_dataset_and_test_config():
    """Load the dataset descriptor and test configuration.
    
    Returns:
        tuple: (dataset_descriptor, test_reserve) where:
            - dataset_descriptor is a list of image pair descriptors
            - test_reserve is a list of image set names reserved for testing
    """
    # Load dataset descriptor containing image metadata
    dataset_descriptor = yaml.safe_load(open(rawproc.RAWNIND_CONTENT_FPATH, 'r'))

    # Load test_reserve configuration to identify test set images
    test_config = yaml.safe_load(open('config/test_reserve.yaml', 'r'))
    test_reserve = test_config['test_reserve']

    return dataset_descriptor, test_reserve


def get_threshold_dict():
    """Create a dictionary of threshold values for MS-SSIM score filtering.
    
    Returns:
        dict: Dictionary with threshold values as keys and empty lists as values
    """
    # Define threshold values from 0.55 to 1.0
    thresholds = [
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
        0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0
    ]

    # Initialize dictionary with empty lists for each threshold
    return {threshold: [] for threshold in thresholds}


def filter_and_group_by_threshold(dataset_descriptor, test_reserve, threshold_dict):
    """Filter images and group them by MS-SSIM score thresholds.
    
    Args:
        dataset_descriptor: List of image pair descriptors from the dataset
        test_reserve: List of image set names reserved for testing
        threshold_dict: Dictionary with threshold values as keys and empty lists
        
    Returns:
        dict: Dictionary with threshold values as keys and lists of MS-SSIM scores
    """
    # Create a copy of the threshold dictionary to store filtered scores
    threshold_scores = threshold_dict.copy()

    # Process each image pair in the dataset
    for image_pair in dataset_descriptor:
        # Skip images that don't meet all filtering criteria:
        # 1. Must be a Bayer pattern image
        # 2. Must have an RGB MS-SSIM score
        # 3. Must belong to the test reserve set
        if (not image_pair['is_bayer'] or
                'rgb_msssim_score' not in image_pair or
                image_pair['image_set'] not in test_reserve):
            continue

        # Print information about the included image
        msssim_score = image_pair['rgb_msssim_score']
        num_crops = len(image_pair['crops'])
        print(f"Image set: {image_pair['image_set']}, "
              f"MS-SSIM: {msssim_score}, Crops: {num_crops}")

        # Group scores by threshold ("less than or equal to" approach)
        for threshold in threshold_scores:
            if msssim_score <= threshold:
                # Add this score once for each crop in the image
                threshold_scores[threshold].extend([msssim_score] * num_crops)

    return threshold_scores


def calculate_statistics(threshold_scores):
    """Calculate statistics for each threshold group.
    
    Args:
        threshold_scores: Dictionary with threshold values and score lists
        
    Returns:
        dict: Dictionary with threshold values and mean scores
    """
    threshold_means = {}

    # Process each threshold group
    for threshold, scores in threshold_scores.items():
        try:
            # Report the number of crops meeting this threshold
            num_crops = len(scores)
            print(f"Threshold {threshold}: {num_crops} crops")

            # Calculate mean score for this threshold group
            if scores:
                threshold_means[threshold] = statistics.mean(scores)
            else:
                print(f"No scores for threshold {threshold}")
                threshold_means[threshold] = None

        except statistics.StatisticsError as e:
            print(f"Error calculating statistics for threshold {threshold}: {e}")

    return threshold_means


def main():
    """Main function to run the MS-SSIM analysis."""
    # Load dataset and test configuration
    dataset_descriptor, test_reserve = load_dataset_and_test_config()

    # Initialize threshold dictionary
    threshold_dict = get_threshold_dict()

    # Filter images and group by threshold
    print("\n===== Filtering Images and Grouping by Threshold =====")
    threshold_scores = filter_and_group_by_threshold(
        dataset_descriptor, test_reserve, threshold_dict
    )

    # Calculate statistics for each threshold
    print("\n===== Calculating Statistics by Threshold =====")
    threshold_means = calculate_statistics(threshold_scores)

    # Print final results
    print("\n===== Final Results: Average MS-SSIM by Threshold =====")
    for threshold in sorted(threshold_means.keys()):
        if threshold_means[threshold] is not None:
            print(f"Threshold {threshold:.2f}: {threshold_means[threshold]:.6f}")

    # Print complete results dictionary
    print("\n===== Complete Results Dictionary =====")
    print(threshold_means)


# Historical results for reference
HISTORICAL_RESULTS = """
Previous analysis results:

Greater than or equal to (ge) thresholds for test reserve:
{0.0: 0.9471623671631659, 0.1: 0.9471623671631659, 0.2: 0.9471623671631659, 
 0.3: 0.9471623671631659, 0.4: 0.9471623671631659, 0.5: 0.9471623671631659, 
 0.55: 0.9528887702900672, 0.6: 0.9528887702900672, 0.65: 0.9577384954803928, 
 0.7: 0.9632890114317769, 0.75: 0.9668514581725878, 0.8: 0.9696365957384678, 
 0.85: 0.9752269836584317, 0.9: 0.9857954731090464, 0.95: 0.9908681560998485, 
 0.96: 0.991699353873151, 0.97: 0.9930015209362592, 0.98: 0.9955494141904637, 
 0.99: 0.9972834067112332, 1.0: 1.0}

Less than or equal to (le) thresholds for test reserve:
{0.55: 0.5386789441108704, 0.6: 0.5386789441108704, 0.65: 0.5752351880073547, 
 0.7: 0.6133408308029175, 0.75: 0.638195092861469, 0.8: 0.664829870685935, 
 0.85: 0.7104434280291848, 0.9: 0.7919600051262475, 0.95: 0.8307980092768931, 
 0.96: 0.8370669360160827, 0.97: 0.8493611401599237, 0.98: 0.8775720073935691, 
 0.99: 0.8977294883411591, 1.0: 0.9471623671631659}

Selected thresholds (le):
0.85: 0.7104434280291848, 0.9: 0.7919600051262475, 
0.97: 0.8493611401599237, 0.99: 0.8977294883411591

Selected thresholds (ge):
0.5: 0.9471623671631659, 0.6: 0.9528887702900672, 0.7: 0.9632890114317769, 
0.8: 0.9696365957384678, 0.9: 0.9857954731090464, 0.95: 0.9908681560998485, 
0.98: 0.9955494141904637, 0.99: 0.9972834067112332, 1.0: 1.0

Training set (ge) thresholds:
{0.0: 0.9466947778591148, 0.1: 0.9466947778591148, 0.2: 0.9466947778591148, 
 0.3: 0.9466947778591148, 0.4: 0.9474856484190777, 0.5: 0.9501742967966526, 
 0.55: 0.953096047711445, 0.6: 0.9566126700179488, 0.65: 0.9617804117395173, 
 0.7: 0.9657521053556828, 0.75: 0.9718392755733448, 0.8: 0.9756409812725337, 
 0.85: 0.9806561555840562, 0.9: 0.986886925951975, 0.95: 0.9922552712003121, 
 0.96: 0.9932532458673509, 0.97: 0.9945093615667424, 0.98: 0.9959005615163854, 
 0.99: 0.9976020557626007, 1.0: 1.0}
"""

if __name__ == "__main__":
    # Run the main analysis
    main()

    # Print historical results for reference
    print("\n")
    print(HISTORICAL_RESULTS)

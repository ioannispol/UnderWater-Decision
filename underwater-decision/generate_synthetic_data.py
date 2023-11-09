import argparse
import random

import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
csv_file_path = '/workspaces/UnderWater-Decision/data/excel/All.csv'
real_data_df = pd.read_csv(csv_file_path)

# Define thresholds for fouling characteristics
HARD_PERCENTAGE_THRESHOLD = 50
HARD_THICKNESS_THRESHOLD = 30
SOFT_PERCENTAGE_THRESHOLD = 50
SOFT_THICKNESS_THRESHOLD = 50

# Constants for depth thresholds and coverage percentages
SHALLOW_DEPTH_UPPER_BOUND = 0
SHALLOW_DEPTH_LOWER_BOUND = -10
MID_DEPTH_UPPER_BOUND = -25
MID_DEPTH_LOWER_BOUND = -35
DEEP_DEPTH_LOWER_BOUND = -40
MAX_COVERAGE_PERCENTAGE = 90
SHALLOW_DEPTH_COVERAGE_RANGE = (5, 11)
DEEP_DEPTH_COVERAGE_RANGE = (70, 91)
GENERIC_COVERAGE_RANGE = (5, 91)


def determine_cleaning_method(hard_perc, hard_mm, soft_perc, soft_mm):
    """
    Determines the cleaning method based on the fouling characteristics.

    Parameters:
    - hard_perc (int): Percentage of hard fouling.
    - hard_mm (int): Thickness of hard fouling in millimeters.
    - soft_perc (int): Percentage of soft fouling.
    - soft_mm (int): Thickness of soft fouling in millimeters.

    Returns:
    - str: The recommended cleaning method.
    """

    # Mechanical cleaning for severe hard fouling
    if hard_perc >= 75 or hard_mm >= 50:
        return 'Mechanical cleaning methods'

    # High-pressure water jetting for severe soft fouling
    if soft_perc >= 75 or soft_mm >= SOFT_THICKNESS_THRESHOLD:
        return 'High-pressure water jetting'

    # Cavitation water jetting for significant hard and soft fouling
    if hard_perc >= HARD_PERCENTAGE_THRESHOLD and soft_perc >= SOFT_PERCENTAGE_THRESHOLD:
        return 'Cavitation water jetting'

    # Ultrasonic cleaning for minor hard fouling without significant thickness
    if hard_perc < HARD_PERCENTAGE_THRESHOLD and hard_mm < HARD_THICKNESS_THRESHOLD:
        return 'Ultrasonic cleaning'

    # Laser cleaning as a default for other cases
    return 'Laser cleaning'


def area_coverage_by_fouling_and_depth(hard_perc: int, soft_perc: int, depth: int) -> int:
    """
    Calculate the area coverage by fouling at a given depth.

    Args:
    hard_perc (int): The percentage of hard fouling.
    soft_perc (int): The percentage of soft fouling.
    depth (int): The depth in meters (negative for below sea level).

    Returns:
    int: The fouling coverage percentage.
    """
    # Handle shallow depths with less coverage due to wave action and cleaning
    if SHALLOW_DEPTH_LOWER_BOUND <= depth <= SHALLOW_DEPTH_UPPER_BOUND:
        return random.randint(*SHALLOW_DEPTH_COVERAGE_RANGE)

    # Handle mid-range depths with the highest fouling thickness
    if MID_DEPTH_LOWER_BOUND <= depth <= MID_DEPTH_UPPER_BOUND:
        return min(max(hard_perc, soft_perc), MAX_COVERAGE_PERCENTAGE)

    # Handle deeper layers with higher coverage, but less than the peak range
    if depth >= DEEP_DEPTH_LOWER_BOUND:
        return random.randint(*DEEP_DEPTH_COVERAGE_RANGE)

    # Default case for other depths
    return min(max(hard_perc, soft_perc), random.randint(*GENERIC_COVERAGE_RANGE))


# Modify the synthetic data generation function to ensure all growth values are integers
def generate_synthetic_data(real_df, num_entries):
    synthetic_data = []
    platforms = real_df['platform'].unique()
    years = real_df['year'].unique()
    depthmins = real_df['depthmin'].dropna().astype(int).unique()
    depthmaxs = real_df['depthmax'].dropna().astype(int).unique()
    items = real_df['Item'].unique()

    for _ in range(num_entries):
        platform = np.random.choice(platforms)
        year = int(np.random.choice(years))
        depthmin = np.random.choice(depthmins)
        depthmax = np.random.choice(depthmaxs)
        item = np.random.choice(items)
        hard_perc = np.random.randint(0, 101)
        hard_mm = np.random.randint(0, int(real_df['hardmm'].dropna().max()) + 1)
        soft_perc = np.random.randint(0, 101 - hard_perc)
        soft_mm = np.random.randint(0, int(real_df['softmm'].dropna().max()) + 1)
        # Use the average depth for coverage calculation
        avg_depth = (depthmin + depthmax) // 2
        total_area_coverage = area_coverage_by_fouling_and_depth(hard_perc, soft_perc, avg_depth)

        cleaning_method = determine_cleaning_method(hard_perc, hard_mm, soft_perc, soft_mm)

        synthetic_data.append({
            "platform": platform,
            "year": year,
            "depthmin": depthmin,
            "depthmax": depthmax,
            "item": item,
            "hardPerc": hard_perc,
            "hardmm": hard_mm,
            "softPerc": soft_perc,
            "softmm": soft_mm,
            "Total_Area_Coverage": total_area_coverage,
            "Recommended_Cleaning_Method": cleaning_method
        })

    return pd.DataFrame(synthetic_data)


def get_args():
    """
    Parse and return command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset based on real data.')
    parser.add_argument('--csv_file_path',
                        type=str,
                        default='/workspaces/UnderWater-Decision/data/excel/All.csv',
                        help='The path to the input CSV file with real data. Defaults to the specified path.')
    parser.add_argument('--synthetic_dataset_path', '-s',
                        type=str,
                        default='/workspaces/UnderWater-Decision/data/default_synthetic_dataset.csv',
                        help='The path where the synthetic dataset CSV will be saved.')
    parser.add_argument('--num_entries', '-n',
                        type=int,
                        default=100,
                        help='The number of entries to generate for the synthetic dataset.')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    csv_file_path = args.csv_file_path
    synthetic_dataset_path = args.synthetic_dataset_path
    num_entries = args.num_entries

    real_data_df = pd.read_csv(csv_file_path)

    synthetic_dataset = generate_synthetic_data(real_data_df, num_entries)

    synthetic_dataset.to_csv(synthetic_dataset_path, index=False)

import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Load the attached Excel file to use as a reference for synthesizing the dataset
excel_data = pd.read_excel("data/mf-data.xlsx")

# Define the parameters for the synthetic dataset based on common attributes and variations
structures = ['Oil Rig', 'Wind Turbine', 'Underwater Pipeline', 'Ship Hull']
locations = ['North Sea', 'Gulf of Mexico', 'Baltic Sea', 'Pacific Ocean']
cleaning_methods = ['Method A', 'Method B', 'Method C']

# Define a function to generate a random date for last cleaning
def generate_random_date(start_year=2015, end_year=2021):
    start_date = datetime(year=start_year, month=1, day=1)
    end_date = datetime(year=end_year, month=1, day=1)
    return (start_date + timedelta(days=random.randint(0, (end_date - start_date).days))).date()

# Function to generate synthetic dataset
def generate_synthetic_data(num_entries):
    synthetic_data = []

    for _ in range(num_entries):
        structure_id = f"{random.randint(1, 999):03}"
        structure_type = random.choice(structures)
        location = random.choice(locations)
        age = random.randint(1, 20)  # Structures aged between 1 to 20 years
        last_cleaning = generate_random_date()
        image_path = f"/path/img{structure_id}"
        detected_algae = random.randint(0, 60)  # Algae coverage between 0% to 60%
        detected_barnacles = random.randint(0, 40)  # Barnacle coverage between 0% to 40%
        detected_mussels = random.randint(0, 30)  # Mussel coverage between 0% to 30%
        total_coverage = detected_algae + detected_barnacles + detected_mussels
        recommended_cleaning_method = random.choice(cleaning_methods)

        # Ensure total coverage doesn't exceed 100%
        total_coverage = min(total_coverage, 100)

        synthetic_data.append({
            "Structure_ID": structure_id,
            "Structure_Type": structure_type,
            "Location": location,
            "Age (years)": age,
            "Last_Cleaning": last_cleaning,
            "Image_Path": image_path,
            "Detected_Algae (%)": detected_algae,
            "Detected_Barnacles (%)": detected_barnacles,
            "Detected_Mussels (%)": detected_mussels,
            "Total_Coverage (%)": total_coverage,
            "Recommended_Cleaning_Method": recommended_cleaning_method
        })
    
    return pd.DataFrame(synthetic_data)

# Generate a synthetic dataset with 10 entries
synthetic_dataset = generate_synthetic_data(10)
synthetic_dataset_path = "data/synthetic_dataset.csv"
synthetic_dataset.to_csv(synthetic_dataset_path, index=False)

# Output the path to the generated CSV file
print(synthetic_dataset_path)

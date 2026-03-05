import os
import pandas as pd

def select_patches_from_dataset(csv_path, dataset_root):
    """
    Selects patches from the CloudSEN12 dataset based on a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        dataset_root (str): Root directory of the downloaded CloudSEN12 dataset.

    Returns:
        list: List of paths to the selected patches.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    selected_patches = []

    for _, row in df.iterrows():
        roi = row['ROI']
        sen2_id = row['sen2']
        label_type = row['label_type']

        # Determine the annotation folder based on label_type
        if label_type == 'high':
            annotation_folder = 'high'
        elif label_type == 'scribble':
            annotation_folder = 'scribble'
        else:
            annotation_folder = 'no-label'

        # Construct the path to the patch
        patch_path = os.path.join(
            dataset_root,
            annotation_folder,
            roi,
            sen2_id
        )

        if os.path.exists(patch_path):
            selected_patches.append(patch_path)
        else:
            print(f"Warning: Patch not found at {patch_path}")

    return selected_patches

# Example usage:
if __name__ == "__main__":
    csv_path = "path/to/your/cloudsen12_patches.csv"  # Replace with your CSV path
    dataset_root = "path/to/your/cloudsen12_dataset"   # Replace with your dataset root

    patches = select_patches_from_dataset(csv_path, dataset_root)
    print(f"Found {len(patches)} patches.")
    for patch in patches[:5]:  # Print first 5 for demo
        print(patch)

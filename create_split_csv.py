
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_split_csvs(image_dir, label_dir, output_dir, train_ratio=0.8):
    """
    Creates train.csv and val.csv for dhSegment-torch training.

    Args:
        image_dir (str): Path to the directory containing original images.
        label_dir (str): Path to the directory containing mask images.
        output_dir (str): Path to the directory where CSV files will be saved.
        train_ratio (float): The proportion of the dataset to allocate to the training set.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith("_mask.png")])

    # Create a DataFrame
    df = pd.DataFrame({
        'image': [os.path.join(image_dir, f) for f in image_files],
        'label': [os.path.join(label_dir, f) for f in label_files]
    })

    # Ensure the image and label basenames match up
    df['image_basename'] = df['image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df['label_basename'] = df['label'].apply(lambda x: os.path.basename(x).replace('_mask.png', ''))

    merged_df = pd.merge(df, df, left_on='image_basename', right_on='label_basename', suffixes=('_img', '_lbl'))
    if len(merged_df) != len(image_files):
        print("Error: Mismatch between image and label files. Ensure every image has a corresponding '_mask.png' label.")
        return

    final_df = merged_df[['image_img', 'label_lbl']].rename(columns={'image_img': 'image', 'label_lbl': 'label'})

    # Split the data
    train_df, val_df = train_test_split(final_df, train_size=train_ratio, random_state=42)

    # Save to CSV
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'val.csv')

    train_df.to_csv(train_csv_path, index=False, header=False)
    val_df.to_csv(val_csv_path, index=False, header=False)

    print(f"Successfully created {train_csv_path} ({len(train_df)} entries)")
    print(f"Successfully created {val_csv_path} ({len(val_df)} entries)")

if __name__ == '__main__':
    # --- IMPORTANT: UPDATE THESE PATHS --- #
    project_data_dir = 'data/my_line_seg_proj' # The main folder for your project
    image_directory = os.path.join(project_data_dir, 'images')
    label_directory = os.path.join(project_data_dir, 'labels')
    # --- END OF PATHS TO UPDATE --- #

    # The CSV files will be saved in your main project data directory
    create_split_csvs(image_directory, label_directory, project_data_dir)

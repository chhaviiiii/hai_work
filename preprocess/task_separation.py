import os
import pandas as pd

def task_separation(raw_data_folder: str, output_folder: str):
    """
    Separates raw Tobii data into tasks, based on the 'Task' column in each file.
    
    Args:
        raw_data_folder (str): Path to the folder containing raw Tobii data files.
        output_folder (str): Path where separated task data files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all raw data files
    for file in os.listdir(raw_data_folder):
        if file.endswith('.pkl'):  # We are working with pickle files
            file_path = os.path.join(raw_data_folder, file)
            df = pd.read_pickle(file_path)  # Load the pickle file into a DataFrame

            # Ensure the 'Task' column exists for task separation
            if 'Task' not in df.columns:
                print(f"Task column not found in {file}. Skipping...")
                continue

            # Group data by the 'Task' column and save each task as a separate file
            grouped_data = df.groupby('Task')
            for task, task_data in grouped_data:
                task_file_name = f"{task}_{file}"  # File naming based on task
                task_data.to_pickle(os.path.join(output_folder, task_file_name))
                print(f"Saved task '{task}' data to {task_file_name}")

# Example usage:
# task_separation('/path/to/raw/data', '/path/to/output/folder')

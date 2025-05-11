def get_validity_percentage(df, time_interval: int, threshold: float = 0.75) -> bool:
    """
    Calculates the validity of the task data. A task is considered valid if 
    it meets the specified validity percentage over the given time interval.

    Args:
        df (DataFrame): Data containing the task data.
        time_interval (int): Duration of the task in seconds.
        threshold (float): The minimum proportion of valid gaze data required.

    Returns:
        bool: Whether the task is valid (True) or not (False).
    """
    # Calculate the number of rows expected for the given time interval
    sampling_rate = 120  # 120 Hz sampling rate
    expected_rows = sampling_rate * time_interval

    # Mask rows where all gaze data is invalid (NaN or negative)
    mask_invalid = (df < 0) | (df.isna())
    invalid_rows = mask_invalid.all(axis=1).sum()  # Count invalid rows

    # Calculate the proportion of valid data
    valid_rows = len(df) - invalid_rows
    validity_percentage = valid_rows / len(df)

    return validity_percentage >= threshold

def validity_analysis(input_folder: str, output_folder: str, time_interval: int = 29, validity_threshold: float = 0.75):
    """
    Analyze the validity of tasks in the dataset, removing invalid tasks.
    
    Args:
        input_folder (str): Folder containing task data.
        output_folder (str): Folder to save valid task files.
        time_interval (int): Time window for the task (in seconds).
        validity_threshold (float): Validity threshold for the task (e.g., 0.75 means 75% of data must be valid).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith('.pkl'):
            file_path = os.path.join(input_folder, file)
            df = pd.read_pickle(file_path)

            if get_validity_percentage(df, time_interval, validity_threshold):
                output_path = os.path.join(output_folder, file)
                df.to_pickle(output_path)
                print(f"Saved valid task data: {file}")
            else:
                print(f"Skipping invalid task data: {file}")
                
# Example usage:
# validity_analysis('/path/to/task/data', '/path/to/output/valid_data')

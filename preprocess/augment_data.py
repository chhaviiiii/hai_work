import matplotlib.pyplot as plt

def cyclic_split(df, n=4):
    """
    Splits a DataFrame into 'n' groups in a cyclic (round-robin) manner.

    Args:
        df (DataFrame): The task data to split.
        n (int): The number of splits.

    Returns:
        list: A list of DataFrames representing each split.
    """
    return [df.iloc[i::n].reset_index(drop=True) for i in range(n)]

def create_scanpath(df, output_directory, filename):
    """
    Create a visual scanpath plot from gaze data and save it as a PNG image.

    Args:
        df (DataFrame): The task data to generate the scanpath.
        output_directory (str): Directory where the scanpath image will be saved.
        filename (str): The name of the file to save the image.
    """
    x = df['GazePointX (ADCSpx)']
    y = df['GazePointY (ADCSpx)']

    plt.figure()
    plt.scatter(x, y, s=5)  # Adjust marker size as needed.
    plt.plot(x, y, linewidth=1)
    plt.axis('off')  # Hide axis for a cleaner image.
    
    output_file = os.path.join(output_directory, f"{filename}.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    print(f"Saved scanpath image for {filename}")
    plt.close()

def augment_data(input_directory: str, output_directory: str, n_splits=4):
    """
    Performs data augmentation by cyclic splitting the tasks and generating scanpaths.

    Args:
        input_directory (str): Directory containing valid task data.
        output_directory (str): Directory where augmented data (split files and scanpath images) will be saved.
        n_splits (int): The number of splits to create for cyclic splitting.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file in os.listdir(input_directory):
        if file.endswith('.pkl'):
            file_path = os.path.join(input_directory, file)
            df = pd.read_pickle(file_path)

            # Perform cyclic splitting
            splits = cyclic_split(df, n_splits)
            
            # Save each split as a new pickle file
            for i, split in enumerate(splits):
                split_filename = f"{file[:-4]}_{i}.pkl"
                split.to_pickle(os.path.join(output_directory, split_filename))
                print(f"Saved cyclic split {i} of {file}.")

                # Generate scanpath image for each split
                create_scanpath(split, output_directory, f"{file[:-4]}_{i}")
                
# Example usage:
# augment_data('/path/to/valid/data', '/path/to/output/augmented_data')

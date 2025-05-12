# VTNet Preprocessing Pipeline

This pipeline prepares raw Tobii eye-tracking data for VTNet, a model designed to detect user confusion. The pipeline includes the following steps:

- **Task Separation**: Raw data is separated into different cognitive tasks.
- **Validity Analysis**: Ensures that the data quality meets the required standards for training.
- **Data Augmentation**: Cyclically splits the data, generates scanpaths, and augments the dataset.
- **Training Setup**: Organizes the preprocessed data into cross-validation splits for training.

##  Preprocessing Steps

### 1. Task Separation  
**Script**: `task_separation.py`  
**Function**: `task_separation(raw_data_folder, output_folder)`

Separates raw Tobii data into individual cognitive tasks using `.seg` + `.tsv` files.

- Matches segment timestamps to raw eye-tracking logs
- Saves individual task `.pkl` files named like `ctrl_1_27.pkl`

```bash
python task_separation.py 
```

---

### 2. Validity Analysis  
**Script**: `validity_analysis.py`  
**Function**: `validity_analysis(input_folder, output_folder, min_valid_ratio=0.75)`

Filters data to keep only sequences with at least 75% valid gaze points.

```bash
python validity_analysis.py
```

---

### 3. Data Augmentation  
**Script**: `augment_data.py`  
**Functions**: `cyclic_split()`, `create_scanpath()`, `augment_data()`

Creates augmented sequences and scanpath images.

- Splits each sequence into 4 chunks (cyclic split)
- Saves `.pkl` and `.png` files per chunk

```bash
python augment_data.py 
```

---

### 4. Training Setup  
**Script**: `training_setup.py`  
**Function**: `create_cv_splits(augmented_dir, n_splits, output_dir)`

Creates grouped K-Fold splits based on user/task IDs.

```bash
python training_setup.py 
```

##  Expected Outputs

- `augmented/*.pkl` and `*.png` – split time-series + scanpaths
- `cv_splits/*.pkl` – train/val splits per fold
- `trained/model_<timestamp>.pt` – trained PyTorch models

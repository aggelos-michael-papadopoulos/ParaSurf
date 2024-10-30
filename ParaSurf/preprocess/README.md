# **Feature Extraction - Preprocessing phase**

This guide outlines the steps needed to generate the ParaSurf 41x41x41x22 input feature vector for training. By following these steps, you will create a dataset ready for training, organized in the specified folder structure.
### Step 1: Clean the Antibody-Antigen Complex
Remove ions, ligands, and water molecules from the antibody-antigen complex and rearrange atom IDs within the PDB structure.
```bash
# Clean the antibody-antigen complex
python clean_dataset.py
```

### Step 2: Sanity Check for Interaction
Verify that at least one antibody heavy atom is within 4.5Å of any antigen heavy atom, ensuring proximity-based interactions.
```bash
# Run sanity check
python check_rec_ant_touch.py
```
### Step 3: Generate Molecular Surface Points
Create the molecular surface for each receptor in the training folder using DMS software. These surface points will serve as a basis for feature extraction.
```bash
# Generate molecular surface points
python create_surfpoints.py
```

### Step 4: Generate ParaSurf Input Feature Grids (41x41x41x22)
```bash
# Create the 3D feature grids for each surface point generated in Step 3. Each feature grid includes 22 channels with essential structural and electrostatic information.
python create_input_features.py
```

### Step 5: Prepare .proteins Files
Generate .proteins files for training, validation, and testing. These files list all receptors (antibodies) to be used in each dataset split.
```bash
# Create train/val/test .proteins files
python create_proteins_file.py
```

### Step 6: Create .samples Files
Generate .samples files, each listing paths to feature files created in Step 4. These files act as a link between features and the training pipeline.
```bash
# Generate .samples files for network training
python create_sample_files.py
```

## **Folder Structure After Preprocessing**


After completing the above steps, the resulting folder structure should be organized as follows:
```bash
├── test_data
│   ├── datasets
│   │   ├── PECAN.samples
│   │   ├── PECAN_TRAIN.proteins
│   │   ├── PECAN_VAL.proteins
│   │   ├── PECAN_TEST.proteins
│   │   └── ...
├── feats
│   ├── PECAN_22
│   ├── Paragraph_Expanded_22
│   └── MIPE_22
├── surfpoints    
│   ├── PECAN
│   │   └── TRAIN
│   ├── Paragraph_Expanded
│   │   └── TRAIN
│   ├── MIPE
│   │   └── TRAIN
└── pdbs # already created from ParaSurf/create_datasets_from_csv
```


Now we are ready for training!
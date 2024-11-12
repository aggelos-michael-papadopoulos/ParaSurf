# **Dataset preperation**

### Steps for Dataset Preparation
`cd ParaSurf/create_datasets_from_csv`
#### Step 1 
Download Specific PDB Files Use the process_csv_dataset.py script to download the PDB files listed in the .csv files.
```bash
# Step 1: Download specified PDB files
python process_csv_dataset.py
```

#### Step 2 
Generate Final Complexes Run final_dataset_preparation.py to arrange the files into complexes with the specified chain IDs from the .csv files.
```bash
# Step 2: Organize files into final complexes
python final_dataset_preparation.py
```


After running these scripts, you will find a test_data/pdbs folder organized as follows:
```bash
├── PECAN
│   ├── TRAIN
│   │   ├── 1A3R_receptor_1.pdb
│   │   ├── 1A3R_antigen_1_1.pdb
│   │   ├── ...
│   │   ├── 5WUX_receptor_1.pdb
│   │   └── 5WUX_antigen_1_1.pdb
│   ├── VAL
│   └── TEST
├── Paragraph_Expanded
│   ├── TRAIN
│   ├── VAL
│   └── TEST
└── MIPE    
    ├── TRAIN_VAL
    └── TEST
```
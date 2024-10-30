# **Feature Extraction - Preprocessing phase**

Steps to create the ParaSurf 41x41x41x22 input feature vector.
### Step 1: Remove ions, ligands and water from the receptor-antibody and then rearrange the atom ids in the PDB structure
```bash
# cleaning the antibody-antigen complex
python clean_dataset.py
```

### Step 2: Sanity check. Ensure at least 1 antibody heavy atom is within 4.5Å of any antigen heavy atom.
```bash
# sanity check
python check_rec_ant_touch.py
```
### Step 3: Create the training molecular surface for each receptor in the training folder.
```bash
# sanity check
python check_rec_ant_touch.py
```

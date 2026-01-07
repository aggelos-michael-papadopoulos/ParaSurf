<p align="center">
  <img src="images/logo.png" alt="ParaSurf Logo" width="325">
</p>

# **ParaSurf**
## **[A Surface-Based Deep Learning Approach for Paratope-Antigen Interaction Prediction](https://doi.org/10.1093/bioinformatics/btaf062)**

ParaSurf is a state-of-the-art surface-based deep learning model for predicting interactions between antibodies and antigens, with outstanding results across three major antibody-antigen benchmarks.


[![License](https://img.shields.io/github/license/aggelos-michael-papadopoulos/ParaSurf)](https://github.com/aggelos-michael-papadopoulos/ParaSurf/blob/main/LICENSE)
[![Paper](https://img.shields.io/badge/Bioinformatics-ParaSurf-blue)](https://doi.org/10.1093/bioinformatics/btaf062)
[![Forks](https://img.shields.io/github/forks/aggelos-michael-papadopoulos/ParaSurf)](https://github.com/aggelos-michael-papadopoulos/ParaSurf/network/members)
[![Stars](https://img.shields.io/github/stars/aggelos-michael-papadopoulos/ParaSurf)](https://github.com/aggelos-michael-papadopoulos/ParaSurf/stargazers)


---
🚀 **Try [ParaSurf](https://huggingface.co/spaces/angepapa/ParaSurf) on 🤗 Hugging Face now!!!**  
**Explore real-time predictions and visualize model outputs interactively!**

<p align="center">
  <img src="images/HF_demo_.gif" alt="ParaSurf Logo" width="800">
</p>

**Note:** If you see *"This Space is sleeping due to inactivity"*, click "Restart this Space" and wait about 3 minutes for it to fully load.

---
### ParaSurf Integration in AI Startups:

![Superbio Logo](images/superbio_logo.png)
**[ParaSurf is live on Superbio.ai!](https://app.superbio.ai/apps/67c5d70367f35115b8de2693)** – Try it out now!

![Neurosnap Logo](images/neurosnap_logo.png)
**[ParaSurf is live on Neurosnap.ai!](https://neurosnap.ai/service/ParaSurf)** – Try it out now!

![Tamarind Logo](images/tamarind_bio_logo.png)
**[ParaSurf is live on tamarind.bio!](https://app.tamarind.bio/tools/parasurf)** – Try it out now!

![ViciBio Logo](images/vici_bio_logo.jpg)
**[ParaSurf is live on Vici.bio!](https://www.vici.bio/parasurf)** – Try it out now!

![proteinIQ Logo](images/proteiniq_logo.png)
**[ParaSurf is live on proteinIQ.io!](https://proteiniq.io/app/parasurf)** – Try it out now!

 ---

**ParaSurf** Results on benchmark datasets:
* PECAN 
* Paragraph Expanded
* MIPE

![Model Results](images/results.jpg)


## **ParaSurf Graphical Abstract**
![ParaSurf](images/ParaSurf.jpg)
![Architecture](images/model%20architecture.jpg)


## Installation

Setup the **Environment**
```bash
# Clone ParaSurf repository
git clone https://github.com/aggelos-michael-papadopoulos/ParaSurf.git 
conda create -n ParaSurf python=3.10 -y
conda activate ParaSurf
cd ParaSurf
pip install -r requirements.txt

# Install OpenBabel for chemical data processing
conda install -c conda-forge openbabel -y
```

Add ParaSurf to PYTHONPATH
```bash
nano ~/.bashrc  
export PYTHONPATH=$PYTHONPATH:/your/path/to/ParaSurf  # change the path to yours
source ~/.bashrc  
```
Install the **DMS software** for the surface molecular representation.
```bash
cd dms
sudo make install
cd ..
```

## 🐳 Docker installation
For detailed instructions on Docker Installation follow [ParaSurf Docker_Installation](Docker/README.md).
## **ParaSurf Model Weights**

All models are hosted [here](https://drive.google.com/drive/folders/1Kpehru9SnWsl7_Wq93WuI_o7f8wrPgpI?usp=drive_link).

Download the best model weights for each benchmark dataset below:


| Dataset                                              | Weights                                                                                                                                                      |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PECAN Dataset                                        | [Download Best Weights](https://drive.google.com/uc?export=download&id=1vZGH-T6K5_ShVma3dwLkLdkoivs09rSP)                                                    |
| Paragraph Expanded                                   | [Download Best Weights](https://drive.google.com/uc?export=download&id=1nd3npYK303e8owDBvW8Ygd5m9SD1puhR)                                                    |
| Paragraph Expanded (Heavy Chains Only)               | [Download Best Weights](https://drive.google.com/uc?export=download&id=16LA99tPYP7vkKpc-ycn98esEUhXUDc-n)                                                    |
| Paragraph Expanded (Light Chains Only)               | [Download Best Weights](https://drive.google.com/uc?export=download&id=1mEBLPKi1sny-inr1ogdYWoo44XgbH8db)                                                    |
| MIPE Dataset (5-fold CV)                             | [best fold-1](https://drive.google.com/uc?export=download&id=1vIg9m557yiQdYsDelbR39Ch2QTKeLJp5), [best fold-2](https://drive.google.com/uc?export=download&id=1wU--r9sMxdF32nmdp3x9N1Ah8UriZVVD), [best fold-3](https://drive.google.com/uc?export=download&id=1LeATakZ-fxIaovQ8wydZbBNfSkbTsub7), [best fold-4](https://drive.google.com/uc?export=download&id=1dj482apCA09sBw2OUQrNR3PVBLEU9okp), [best fold-5](https://drive.google.com/uc?export=download&id=1Ai0VnytiUNUJsJ5ifKuBv4nkiz35iI6M) |



## **Blind antibody binding site prediction**
Run a blind prediction on a receptor file to identify binding sites:

```bash
python blind_predict.py --receptor test_blind_prediction/4N0Y_receptor_1.pdb --model_weights path/to/model_weights
```
![Alt text](images/pred_example.png)


## Create Datasets and train ParaSurf from scratch
Prepare the dataset from initial .csv files and create ParaSurf features for training.

### 1. Create dataset from the .csv files
For detailed instructions on preparing the dataset, see the [Dataset Preparation README](ParaSurf/create_datasets_from_csv/README.md).

### 2. Feature extraction
Follow instructions for feature extraction in the [ParaSurf Feature Extraction](ParaSurf/preprocess/README.md).

### 3. Train ParaSurf
Train and validate ParaSurf to reproduce the paper's results:
```bash
# Train ParaSurf model
python ParaSurf/train/train.py
```

## Reproduce ParaSurf Results
Once you have the best weights (downloaded or created from scratch), you can validate the model on the test sets and reproduce the results presented in the ParaSurf paper:

```bash
# Validate the ParaSurf model on the TEST set 
python ParaSurf/train/validation.py

# Reproduce ParaSurf results
python ParaSurf/train/Post_process_validation_resutls.py
```

# Contacts
For any issues or questions, please don't hesitate to reach out: [angepapa@iti.gr](mailto:angepapa@iti.gr)

# License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.


# 📍 Citation
If you use ParaSurf, please cite:
```bibtex
@article{papadopoulos2025parasurf,
  title={ParaSurf: A Surface-Based Deep Learning Approach for Paratope-Antigen Interaction Prediction},
  author={Papadopoulos, Angelos-Michael and Axenopoulos, Apostolos and Iatrou, Anastasia and Stamatopoulos, Kostas and Alvarez, Federico and Daras, Petros},
  journal={Bioinformatics},
  pages={btaf062},
  year={2025},
  publisher={Oxford University Press}
}
```

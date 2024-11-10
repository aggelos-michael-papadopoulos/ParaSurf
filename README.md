# **ParaSurf**
## **A Surface-Based Deep Learning Approach for Paratope-Antigen Interaction Prediction**

ParaSurf is a state-of-the-art surface-based deep learning model for predicting interactions between paratopes and antigens, with outstanding results across three major antibody-antigen benchmarks:

* PECAN 
* Paragraph Expanded
* MIPE

![Alt text](images/results.jpg)


## **ParaSurf Graphical Absract**
![Alt text](images/ParaSurf.jpg)
![Alt text](images/model%20architecture.jpg)


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

## **ParaSurf Model Weights**

All models are hosted [here](https://drive.google.com/drive/folders/1Kpehru9SnWsl7_Wq93WuI_o7f8wrPgpI?usp=drive_link).

Best model weights for the 3 benchmark datasets:


| Dataset                                | Weights                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PECAN Dataset                          | [Download best Weights](https://drive.google.com/uc?export=download&id=1vZGH-T6K5_ShVma3dwLkLdkoivs09rSP)                                                                                                                                                                                                                                                                                                                                                                                           |
| Paragraph Expanded                     | [Download best Weights](https://drive.google.com/uc?export=download&id=1nd3npYK303e8owDBvW8Ygd5m9SD1puhR)                                                                                                                                                                                                                                                                                                                                                                                           |
| Paragraph Expanded (Heavy Chains Only) | [Download best Weights](https://drive.google.com/uc?export=download&id=16LA99tPYP7vkKpc-ycn98esEUhXUDc-n)                                                                                                                                                                                                                                                                                                                                                                                           |
| Paragraph Expanded (Light Chains Only) | [Download best Weights](https://drive.google.com/uc?export=download&id=1mEBLPKi1sny-inr1ogdYWoo44XgbH8db)                                                                                                                                                                                                                                                                                                                                                                                           |
| MIPE Dataset (5-fold CV)               | [best fold-1](https://drive.google.com/uc?export=download&id=1vIg9m557yiQdYsDelbR39Ch2QTKeLJp5), [best fold-2](https://drive.google.com/uc?export=download&id=1wU--r9sMxdF32nmdp3x9N1Ah8UriZVVD), [best fold-3](https://drive.google.com/uc?export=download&id=1LeATakZ-fxIaovQ8wydZbBNfSkbTsub7), [best fold-4](https://drive.google.com/uc?export=download&id=1dj482apCA09sBw2OUQrNR3PVBLEU9okp), [best fold-5](https://drive.google.com/uc?export=download&id=1Ai0VnytiUNUJsJ5ifKuBv4nkiz35iI6M) |


## **Blind paratope binding site prediction**
Run blind binding-site prediction.

```bash
python blind_predict.py --receptor "test_blind_prediction/4N0Y_receptor_1.pdb" --model_weights "path/to/model_weights"
```
![Alt text](images/pred_example.png)


## Create Dataset from scratch
Prepare the dataset from initial .csv files and create ParaSurf features for training.

### 1. Create dataset from the .csv files
For detailed instructions on preparing the dataset, see the [Dataset Preparation README](ParaSurf/create_datasets_from_csv/README.md).

### 2. Feature extraction
Follow instructions for feature extraction in the [ParaSurf Feature Extraction](ParaSurf/preprocess/README.md).

### 3. Train ParaSurf
Train and validate ParaSurf to reproduce the paper's results:
```bash
# Train and validate ParaSurf model
python ParaSurf/train/train.py

# Validate to reproduce paper results
python ParaSurf/train/validation.py
```

## **Try ParaSurf on Hugging Face!** 🤗


We’re excited to share that **ParaSurf** is now live on Hugging Face 🤗! Try it out [here](https://huggingface.co/spaces/angepapa/ParaSurf).

# TODO:
- [x] Upload final weights to repository
- [x] Upload requirements file for the ParaSurf environment
- [x] Create web-demo at Hugging face

# Citation
coming soon ...

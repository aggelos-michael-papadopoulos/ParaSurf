# **ParaSurf**
## **Surface-Based Deep Learning for Paratope-Antigen Interaction Prediction**

ParaSurf is a state-of-the-art surface-based deep learning model for predicting interactions between paratopes and antigens, with outstanding results across three major antibody-antigen benchmarks:

* PECAN 
* Paragraph Expanded
* MIPE

![Alt text](images/results.png)


## **ParaSurf Graphical Absract**
![Alt text](images/graphical_abstract.jpg)


## INSTALL
Install the **DMS software** for the surface molecular representation.
```bash
cd dms
sudo make install
```

Install **ParaSurf**
```bash
# Install enviroment
git clone https://github.com/aggelos-michael-papadopoulos/ParaSurf.git 
conda create -n ParaSurf python=3.9
conda activate ParaSurf

# Install the dependencies
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torchsummary scipy tqdm h5py jsonpickle pandas biopython scikit-learn matplotlib wandb
conda install -c conda-forge openbabel
conda install numpy=1.24
```




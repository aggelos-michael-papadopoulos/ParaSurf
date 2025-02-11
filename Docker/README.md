# **Docker Installation 🐳**

## 🚀 Build the Docker Environment
```bash
sudo docker build -t parasurf:latest .
```
## 🏃‍♂️ Run the Docker Container
```bash
sudo docker run --gpus all -it --rm \
    --name parasurf_container \
    -v $(pwd):/workspace \
    parasurf:latest
```

## 📥 Download Model Weights

```bash
# Activate ParaSurf
conda activate ParaSurf
# Download model weights
mkdir -p /workspace/weights && \
    conda run -n ParaSurf gdown --id 1LBydgQ7sTXTAuEdE3Le_PH6cY_F2h_Tp -O /workspace/weights/ParaSurf_best.pth && \
    conda run -n ParaSurf gdown --id 1vZGH-T6K5_ShVma3dwLkLdkoivs09rSP -O /workspace/weights/Pecan_best.pth && \
    conda run -n ParaSurf gdown --id 1nd3npYK303e8owDBvW8Ygd5m9SD1puhR -O /workspace/weights/Paragraph_expanded_best.pth
```

## Run a prediction 🔍
```bash
python blind_predict.py --receptor test_blind_prediction/4N0Y_receptor_1.pdb --model_weights weights/ParaSurf_best.pth
```
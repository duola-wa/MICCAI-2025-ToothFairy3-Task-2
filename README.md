# MICCAI-2025-ToothFairy3-Task-2

## nnU-Net based Model:
### 1. nnUNet Configuration
Install nnUNetv2 as below.  
You should meet the requirements of nnUNetv2, our method does not need any additional requirements.  
For more details, please refer to https://github.com/MIC-DKFZ/nnUNet  

```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
### 2. Dataset

Load ToothFairy3 Dataset from https://toothfairy3.grand-challenge.org/dataset/

### 3. Preprocessing

Conduct automatic preprocessing using nnUNet.

```
nnUNetv2_plan_and_preprocess -d 302 --verify_dataset_integrity
```


### 4. Training

Train by nnUNetv2. 

Run script:

```
nnUNetv2_train 302 3d_fullres all -tr nnUNetTrainerNoDA
```


### 5. Inference

Test by nnUNetv2 with post-processing. 

Run script:

```
python nnUnet-based-method/test_3D.py
```

## nnInteractive based Model:
### 1. nnInteractive Configuration
Install nnInteractive as below.  
```
git clone https://github.com/MIC-DKFZ/nnInteractive
cd nnInteractive
pip install -e .
```
For more details, please refer to https://github.com/MIC-DKFZ/nnInteractive/tree/master  

### 2. Dataset

Load ToothFairy3 Dataset from https://toothfairy3.grand-challenge.org/dataset/

### 3. Download Pre-trained Model

Load  Pre-trained Model from https://huggingface.co/nnInteractive/nnInteractive/tree/main

### 4. Inference

Test by nnInteractive. 

Run script:

```
python nnInteractive-based-method/test_nninteractive.py
```



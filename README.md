# SILC
This is the repository of our work "Multi-grained Radiology Report Generation with Sentence-level Image-language Contrastive Learning".
Some files are unavailable on github due to file size restrictions. You can download our project (including model checkpoint) from https://drive.google.com/drive/folders/1Oax4RYRaZakV3CbAFVhHQbDqucFXTTFv?usp=sharing.
## Requirements

- `torch==1.9.0`
- `tensorboard==1.15.0`
- `torchvision==0.10.0`
## Dataset
We provide the IU X-ray dataset in `./dataset/IU`. The images are preprocessed and have lower resolutions. The MIMIC-CXR dataset is too big and you can download it [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and put it in `./dataset/MIMIC`.

The annotations are prepared by us and stored in `./preprocess/IU` and `./preprocess/MIMIC`.

## Models
The trained models are stored in `./checkpoint`.

## Evaluation
Before evaluation, please prepare the evaluation tool. You can download [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) and put it in `./pycocoevalcap` (we have prepared it). Please also install Java. 

Run `bash evaluate_IU.sh` to evaluate on IU X-Ray dataset and `bash evaluate_MIMIC.sh` to evaluate on MIMIC-CXR dataset. The result will be stored in `./outputs`

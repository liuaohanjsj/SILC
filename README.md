# SILC
This is the repository of our work "Multi-grained Radiology Report Generation with Sentence-level Image-language Contrastive Learning".
## Requirements

- `torch==1.9.0`
- `tensorboard==1.15.0`
- `torchvision==0.10.0`
## Dataset
We provide the IU X-ray dataset in ./dataset/IU. The images are preprocessed and have lower resolutions. The MIMIC-CXR dataset is too big and you can download it [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
The annotations are prepared by us and stored in ./preprocess/IU and ./preprocess/MIMIC.
## Evaluation
Before evaluation, please prepare the evaluation tool. You can download [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) and put it in `./pycocoevalcap`. Please also install Java. 
Run `bash evaluate_IU.sh` to evaluate on IU X-Ray dataset and `bash evaluate_MIMIC.sh` to evaluate on MIMIC-CXR dataset.

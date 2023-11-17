# BKAI-IGH NeoPolyp - Deep learning Assignment 3

Name: Nguyen Trung Truc
Student ID: 20214936


Step 1:
Add  "bkai-igh-neopolyp" dataset to /kaggle/input/

Step 2:
First, we need to download the "unet_model.pth" from Google Drive and put it in "/kaggle/working/"

```python
!pip install  gdown
import gdown

url =

output = '/kaggle/working/unet_model.pth'
gdown.download(url, output)
```

Step 3:

```python
!git clone https://github.com/cristiano2003/DL-Assignment-3.git
%cd /kaggle/working/UnetPolyp-Semantic-Segmentation/
!pip install -r requirements.txt
!pip install segmentation_models_pytorch
!python infer.py
```

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

url = 'https://drive.google.com/uc?id=10ZtAN1DTSnlq8rNjhV_6zVIzl0v_qy1R&export=download'

output = '/kaggle/working/unet_model.pth'
gdown.download(url, output)
```

Step 3:

```python
!git clone https://github.com/cristiano2003/UnetPolyp-Semantic-Segmentation.git
%cd /kaggle/working/UnetPolyp-Semantic-Segmentation/
!pip install -r requirements.txt
!pip install segmentation_models_pytorch
!python infer.py
```

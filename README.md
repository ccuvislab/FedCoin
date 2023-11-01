# A Privacy-preserving Approach for Multi-source Domain Adaptive Object Detection

![architecture](pic/architecture.png)

This repo is the official implementation of ICIP paper "[A Privacy-preserving Approach for Multi-source Domain Adaptive Object Detection](https://ieeexplore.ieee.org/document/10222121/)" by Peggy Lu, Chia-Yung Jui, Jen-Hui Chuang.

## Installation

### Prerequisites
```shell
pip install -r requirements.txt
```

### Install Detectron2
Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2. We use version: ```detectron2==0.5```

## Usage

### Data Preparation
Plz refer to [prepare_data.md](docs/prepare_data.md) for datasets preparation.

### Pretrained Model

We used VGG16 pre-trained on ImageNet for all experiments. You can download it to ```/path/to/project```:

- VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

### Training

### main file
* train_net_FedAvg.py
* train_net_FedMA.py
* train_net_multiTeacher.py

### config
You can change ```--config configs/pt/final_c2f.yaml``` to other configs in ```configs/xxx``` to reproduce the main results of other tasks.

* configs/FedAvg/xxx.yaml
* configs/FedMA/xxx
* configs/multi-teacher
* configs/source-only


##  training
```
python train_net_multiTeacher.py --config-file configs/multi-teacher/skf2c_foggy_sourceonly_FedMAbackbone.yaml
```


## Main Results (to be modified)
This code has been further improved,  achiving more superior adaptation performance than the results presented in the paper (about +1~2 mAP gains across the tasks, see exps logs for details).
|Adaptation Tasks |Methods|Model Weights | mAP50                 | Log |
| ---- | -------| ----- |------------------------------|------------------------------|
|CitysScape2FoggyCityscape| PT (ours) | [Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing) | **31 &rArr; 47.1 (+16.1)**   | [Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing) |
|CitysScape2BDD100k| PT (ours) | [Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing) | **26.9 &rArr; 34.9 (+8.0)**  |[Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing)  |
|KITTI2CitysScape| PT (ours) | [Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing) | **46.4 &rArr; 60.2 (+13.8)** | [Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing) |
|Sim10k2CitysScape|PT (ours) | [Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing) | **44.5 &rArr; 55.1 (+10.6)** | [Google Drive](https://drive.google.com/drive/folders/1rMXAaJpgOOHycnGhL2RwLJaz6dspmb74?usp=sharing) |


## Citation

If you use this project in your research or wish to refer to the results published in the paper, please consider citing our paper:
```BibTeX
@INPROCEEDINGS{10222121,
  author={Lu, Peggy Joy and Jui, Chia-Yung and Chuang, Jen-Hui},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)}, 
  title={A Privacy-Preserving Approach for Multi-Source Domain Adaptive Object Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1075-1079},
  doi={10.1109/ICIP49359.2023.10222121}}
```

## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.

## Acknowledgement

This project is built upon [Detectron2](https://github.com/facebookresearch/detectron2) and [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher), and we'd like to appreciate for their excellent works.

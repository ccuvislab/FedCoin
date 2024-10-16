# (FedCoin) Federated Contrastive Domain Adaptation for Category-inconsistent Object Detection

![architecture](pic/architecture.png)

This repo is the official implementation of VCIP paper "[Federated Contrastive Domain Adaptation for Category-inconsistent Object Detection](https://ieeexplore.ieee.org/document/xxxx/)" by WeiYu Chen, Peggy Lu, Vincent S. Tseng.

# Installation

This requirements is including the project Detectron2 [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). We use version: ```detectron2==0.6```

```shell
conda install --file requirements.txt
```

### Data Preparation
Plz refer to [prepare_data.md](docs/prepare_data.md) for datasets preparation.

### Pretrained Model

We used VGG16 pre-trained on ImageNet for all experiments. You can download it to ```/path/to/project```:

- VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

# Usage

### Experiments Setup
![architecture](pic/experiments_setup.png)


### main file
* train_net_FedAvg.py
* train_net_FedMA.py
* train_net_multiTeacher.py

## Train on Client
### FedAvg

```
python train_net_FedAvg.py --config configs/multiclass/avg01_ck2b.yaml
python train_net_FedAvg.py --config configs/multiclass/avg02_skf2c.yaml
python train_net_FedAvg.py --config configs/multiclass/avg03_ck2b.yaml
python train_net_FedAvg.py --config configs/multiclass/avg04_skf2c.yaml
```

### FedMA  

```
python train_net_FedMA.py --config configs/multiclass/ma01_ck2b.yaml
python train_net_FedMA.py --config configs/multiclass/ma02_skf2c.yaml
python train_net_FedMA.py --config configs/multiclass/ma03_ck2b.yaml
python train_net_FedMA.py --config configs/multiclass/ma04_skf2c.yaml
```


## Train on Server

### Multi Target （ 8 class ）
```
python train_net_multiTeacher.py --config configs/multiclass/mt01_avg_ck2b_so.yaml
python train_net_multiTeacher.py --config configs/multiclass/mt02_avg_skf2c_so.yaml
python train_net_multiTeacher.py --config configs/multiclass/mt03_avg_ck2b_moon.yaml
python train_net_multiTeacher.py --config configs/multiclass/mt04_avg_skf2c_moon.yaml
python train_net_multiTeacher.py --config configs/multiclass/mt05_ma_ck2b_so.yaml
python train_net_multiTeacher.py --config configs/multiclass/mt06_ma_skf2c_so.yaml
python train_net_multiTeacher.py --config configs/multiclass/mt07_ma_ck2b_moon.yaml
python train_net_multiTeacher.py --config configs/multiclass/mt08_ma_skf2c_moon.yaml
```


### MultiTeacher [ Ablation Study ] 
```
python train_net_multiTeacher.py  --config configs/ablation/avg03_dyn_ck2b.yaml
python train_net_multiTeacher.py  --config configs/ablation/avg03_inv_ck2b.yaml
python train_net_multiTeacher.py  --config configs/ablation/avg04_dyn_skf2c.yaml
python train_net_multiTeacher.py  --config configs/ablation/avg04_inv_skf2c.yaml
python train_net_multiTeacher.py  --config configs/ablation/mt03_dyn_ck2b.yaml
python train_net_multiTeacher.py  --config configs/ablation/mt03_inv_ck2b.yaml
python train_net_multiTeacher.py  --config configs/ablation/mt04_dyn_skf2c.yaml
python train_net_multiTeacher.py  --config configs/ablation/mt04_inv_skf2c.yaml
```

# Case Study 

### case 1 :  run fedAvg for  skf→c ,  and evaluation only for this model
```
# train __your_FedAvg_skf2c_model, you need to modify avg02_skf2c.yaml
python train_net_FedAvg.py --config configs/multiclass/avg02_skf2c.yaml
# evaluate  __your_FedAvg_skf2c_model.pth
python train_net.py --config configs/evaluation/cityeval.yaml --eval-only MODEL.WEIGHTS output/__your_FedAvg_skf2c_model.pth 
```

### case 2 : run  fedMA to FedCoin with contrastive method for skf → c dataset
```
# train on client by fedMA
python train_net_FedMA.py --config configs/202405_multiclass/ma04_skf2c.yaml

# domain adapt on server with contrastive method
python train_net_multiTeacher.py --config configs/multiclass/mt08_ma_skf2c_moon.yaml
```

### case 3 : run fedAvg to FedCoin with ablation study for ck → b dataset
```
## inverse contrastive method
# train on client by fedAvg with inverse contrastive method
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/ablation/avg03_inv_ck2b.yaml
# domain adapt on server with inverse contrastive method
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/ablation/mt03_inv_ck2b.yaml

## dynamic contrastive method
# train on client by fedAvg with dynamic contrastive method
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/ablation/avg03_dyn_ck2b.yaml
# domain adapt on server with dynamic contrastive method
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/ablation/mt03_dyn_ck2b.yaml
```


# Related

## Citation

If you use this project in your research or wish to refer to the results published in the paper, please consider citing our paper:
```BibTeX
```

## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.

## Acknowledgement

This project is built upon [Detectron2](https://github.com/facebookresearch/detectron2) and [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher), and we'd like to appreciate for their excellent works.

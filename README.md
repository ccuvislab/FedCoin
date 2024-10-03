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

# Training

### main file
* train_net_FedAvg.py
* train_net_FedMA.py
* train_net_multiTeacher.py

### Evaluation only
```shell
python train_net.py --config configs/moon/cityeval.yaml --eval-only MODEL.WEIGHTS output/FedAvg_skf2c_sourceonly_moon_20240123/FedAvg_0.pth 
```
###  多gpu 跑法
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/moon/FedAvg_ck2b_sourceonly.yaml 
```
###  Multi-Teacher  20240611  
 * @g02 一次兩行在跑
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg04_skf2c.yaml; CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_multiclass/mt04_avg_skf2c_moon.yaml
```
 * @g03 分段跑
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_dyn_skf2c.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt04_dyn_skf2c.yaml
```
 * @g02 一次兩行在跑
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_inv_skf2c.yaml; CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt04_inv_skf2c.yaml
```
### Dynamic Moon Pos-Nav in Multi-Teacher  20240607
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg03_dyn_ck2b.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_dyn_skf2c.yaml
```
### Inverse Moon Pos-Nav in Multi-Teacher Average  20240517
 * mt03 inverse moon [ mt03 w/ avg03 ]
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg03_inv_ck2b.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt03_inv_ck2b.yaml
```
 * mt04 inverse moon [ mt04 w/ avg04 ]
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_inverse_moon/avg04_inv_skf2c.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_net_multiTeacher.py  --num-gpus 2 --config configs/202405_inverse_moon/mt04_inv_skf2c.yaml
```
## Multi-Teacher  20240506
```shell
python train_net_multiTeacher.py   --config configs/202405_multiclass/mt07_ma_ck2b_moon.yaml
python train_net_multiTeacher.py   --config con
```
### AVG   20240411
 * Avg01 so ck -> b (so)
```shell
python train_net_FedAvg.py --config configs/202405_multiclass/avg01_ck2b.yaml
```
 * Avg02 so skf -> c (so)
```shell
python train_net_FedAvg.py --config configs/202405_multiclass/avg02_skf2c.yaml
```
 * Avg03 moon ck -> b (moon)
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg03_ck2b.yaml
```
 * Avg04 moon skf -> c (moon)
```shell
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg04_skf2c.yaml
```
### MA 20240411
 * MA01 moon ck -> b （moon）
```shell
python train_net_FedMA.py --config configs/202405_multiclass/ma01_ck2b.yaml
```
 * MA02 moon skf -> c (moon)
```shell
python train_net_FedMA.py --config configs/202405_multiclass/ma02_skf2c.yaml
```
 * MA03 so ck -> b  (so版)
```shell
python train_net_FedMA.py --config configs/202405_multiclass/ma03_ck2b.yaml 
```
 * MA04 so skf -> c (so版)
```shell
python train_net_FedMA.py --config configs/202405_multiclass/ma04_skf2c.yaml # 
```

### single class 20240331 
 * FedAvg sourceonly 
```shell
python train_net_FedAvg.py --config-file configs/FedAvg/FedAvg_skf2c_multiclass.yaml
```
 * FedAvg moon
```shell
python train_net_FedAvg.py --config-file configs/FedAvg/FedAvg_skf2c_multiclass_moon.yaml
```
 * FedMA sourceonly
```shell
python train_net_FedMA.py --config-file configs/FedMA/ck2b_FedMA_8cla.yaml
```
 * FedMA moon
```shell
python train_net_FedMA.py --config-file configs/FedMA/ck2b_FedMA_8cla_moon.yaml
```

## Citation

If you use this project in your research or wish to refer to the results published in the paper, please consider citing our paper:
```BibTeX
```

## License

This project is released under the [Apache 2.0 license](./LICENSE). Other codes from open source repository follows the original distributive licenses.

## Acknowledgement

This project is built upon [Detectron2](https://github.com/facebookresearch/detectron2) and [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher), and we'd like to appreciate for their excellent works.

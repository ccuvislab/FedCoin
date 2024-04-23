##### 20240411

## 多gpu 跑法
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/moon/FedAvg_ck2b_sourceonly.yaml 

## Avg01 so ck -> b
python train_net_FedAvg.py --config configs/202405_multiclass/avg01_ck2b.yaml

## Avg02 so skf -> c
python train_net_FedAvg.py --config configs/202405_multiclass/avg02_skf2c.yaml

## Avg03 moon ck -> b
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg03_ck2b.yaml

## Avg04 moon skf -> c
CUDA_VISIBLE_DEVICES=0,1 python train_net_FedAvg.py --num-gpus 2 --config configs/202405_multiclass/avg04_skf2c.yaml
  
## MA01 so ck -> b
python train_net_FedMA.py --config configs/202405_multiclass/ma01_ck2b.yaml

## MA02 so skf -> c
python train_net_FedMA.py --config configs/202405_multiclass/ma02_skf2c.yaml

## MA03 moon ck -> b
#CUDA_VISIBLE_DEVICES=0,1 python train_net_FedMA.py --num-gpus 2 --config configs/202405_multiclass/ma03_ck2b.yaml

## MA04 moon skf -> c
#CUDA_VISIBLE_DEVICES=0,1 python train_net_FedMA.py --num-gpus 2 --config configs/202405_multiclass/ma04_skf2c.yaml # not ready 
  



#####  20240331 
## FedAvg sourceonly
python train_net_FedAvg.py --config-file configs/FedAvg/FedAvg_skf2c_multiclass.yaml
## FedAvg moon
python train_net_FedAvg.py --config-file configs/FedAvg/FedAvg_skf2c_multiclass_moon.yaml

## FedMA sourceonly
python train_net_FedMA.py --config-file configs/FedMA/ck2b_FedMA_8cla.yaml
## FedMA moon
python train_net_FedMA.py --config-file configs/FedMA/ck2b_FedMA_8cla_moon.yaml


## Evaluation only
python train_net.py --config configs/moon/cityeval.yaml --eval-only MODEL.WEIGHTS output/FedAvg_skf2c_sourceonly_moon_20240123/FedAvg_0.pth 

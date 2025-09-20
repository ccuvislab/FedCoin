# FedProx 使用範例
# 展示如何在統一訓練腳本中使用 FedProx 算法

"""
使用 FedProx 的配置範例：

1. 在配置文件中設置：
FEDSET:
  ALGORITHM: "fedprox"
  MU: 0.01  # Proximal term 係數
  ROUND: 10
  DATASET_LIST: ["VOC2007_citytrain1", "VOC2007_kitti1"]

2. 運行命令：
python train_net_unified.py --config-file configs/fedprox_config.yaml

3. FedProx 的優勢：
   - 更好的收斂穩定性
   - 適合 Non-IID 數據分佈
   - 減少客戶端漂移問題

4. 參數調節建議：
   - MU = 0.01: 輕微穩定性提升（推薦起始值）
   - MU = 0.05: 中等穩定性提升
   - MU = 0.1: 強穩定性，但可能過於保守
"""

# FedProx vs FedAvg 對比範例
def compare_algorithms():
    """
    展示 FedProx 和 FedAvg 的差異
    """
    
    # FedAvg 聚合
    def fedavg_aggregate(client_weights):
        return torch.stack(client_weights, dim=0).mean(dim=0)
    
    # FedProx 聚合  
    def fedprox_aggregate(client_weights, global_weights, mu=0.01):
        avg = torch.stack(client_weights, dim=0).mean(dim=0)
        return (1 - mu) * avg + mu * global_weights
    
    print("FedProx 通過 proximal term 提供更好的穩定性")
    print("特別適合處理異質性數據分佈的聯邦學習場景")




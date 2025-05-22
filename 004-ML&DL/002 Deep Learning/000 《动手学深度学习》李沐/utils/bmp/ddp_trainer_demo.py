import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import AdamW
import ddp_trainer as ddpt

# --- 1. 定义模拟数据集 ---
class FakeImageDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=224, num_classes=10):
        self.data = torch.randn(num_samples, 3, img_size, img_size)  # 模拟图像数据
        self.labels = torch.randint(0, num_classes, (num_samples,))  # 随机标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- 2. 定义简易CNN模型 ---
class DemoModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --- 3. 配置训练参数 ---
config = {
    'model': DemoModel(),
    'train_dataset': FakeImageDataset(num_samples=1000),
    'val_dataset': FakeImageDataset(num_samples=200),
    'optimizer': AdamW(DemoModel().parameters(), lr=1e-4),
    'loss_fn': nn.CrossEntropyLoss(),
    'device': 'cuda',
    'num_epochs': 5,
    'batch_size': 32,
    'save_dir': './ddp_demo',
    'use_amp': True,          # 启用混合精度
    'grad_clip': 1.0,         # 梯度裁剪
    'lr_scheduler': None      # 本示例暂不使用调度器
}

# --- 4. 主函数（启动入口）---
def main(rank, world_size):
    # 初始化分布式训练
    ddpt.setup_ddp(rank, world_size)
    
    # 实例化Trainer
    trainer = ddpt.DDPMultiGPUTrainer(config)
    
    # 开始训练
    trainer.train()
    
    # 清理环境
    ddpt.cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Available GPUs: {world_size}")
    
    # 启动多进程训练
    torch.multiprocessing.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
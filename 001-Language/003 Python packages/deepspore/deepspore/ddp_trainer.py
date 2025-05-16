import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# 1. 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# 2. 极简训练器
class DDPTrainer:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.model = nn.Linear(10, 2).to(rank)  # 示例模型
        self.ddp_model = DDP(self.model, device_ids=[rank])
        self.optimizer = torch.optim.SGD(self.ddp_model.parameters(), lr=0.01)
        
        # 模拟数据
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 10), 
            torch.randint(0, 2, (1000,))
        )
        
        # 分布式数据加载
        self.sampler = DistributedSampler(
            self.dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=32,
            sampler=self.sampler,
            num_workers=4,
            pin_memory=True
        )

    def train_epoch(self, epoch):
        self.ddp_model.train()
        self.sampler.set_epoch(epoch)
        
        for data, target in self.loader:
            data = data.to(self.rank, non_blocking=True)
            target = target.to(self.rank, non_blocking=True)
            
            self.optimizer.zero_grad()
            output = self.ddp_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            self.optimizer.step()
            
            if self.rank == 0:  # 仅主进程打印
                print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    def save(self, path):
        if self.rank == 0:  # 仅主进程保存
            torch.save(self.ddp_model.module.state_dict(), path)

# 3. 启动函数
def main(rank, world_size):
    setup(rank, world_size)
    
    trainer = DDPTrainer(rank, world_size)
    for epoch in range(3):  # 训练3个epoch
        trainer.train_epoch(epoch)
    
    trainer.save("ddp_model.pt")
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
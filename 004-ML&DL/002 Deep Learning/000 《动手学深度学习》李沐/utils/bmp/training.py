import random 
import pickle 
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from collections import defaultdict
from IPython import display
from tqdm import tqdm 


# Function for setting the seed
def set_seed(seed: int = 42)-> None:
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set seed {seed} for reproducibility.")

    return None


class MetricTracker:
    def __init__(self):
        '''
        # 示例输出结构
        history = {
            'epoch': {
                'train_loss_epoch': [0.5, 0.4, 0.3],          # 每个epoch的指标
                'train_acc_epoch': [0.8, 0.85, 0.9],
                'val_loss_epoch': [0.6, 0.5, 0.4],
                'val_acc_epoch': [0.7, 0.75, 0.8]
            },
            'step': {
                'train_loss_step': [0.55, 0.45, 0.35],        # 每个step的指标
                'train_acc_step': [0.78, 0.83, 0.88],
                'val_loss_step': [0.62, 0.52, 0.34],          # 验证阶段一般只在epoch结束时计算
                'val_acc_step': [0.72, 0.77, 0.77],
            },
        }     
        '''
        self._metrics = {}                          # 存储指标计算函数
        self._epoch_buffer = defaultdict(list)      # Epoch级别累积
        self._step_buffer = defaultdict(list)       # Step级别累积
        self._history = {
            'epoch': defaultdict(list),             # 按阶段和指标名存储epoch指标
            "step": defaultdict(list)               # 按阶段和指标名存储step指标
        }
        self.current_stage = 'train'                # 当前阶段标识

    def add_metric(self, name, metric_fn):
        """注册指标（如损失、准确率）"""
        self._metrics[name] = metric_fn

    def set_stage(self, stage):
        """设置当前阶段（train/val/test）"""
        self.current_stage = stage

    def update(self, **kwargs):
        """更新缓冲区（需传入指标函数所需的参数），紧邻每个batch之后计算。"""
        for name, fn in self._metrics.items():
            value = fn(**kwargs)
            self._epoch_buffer[name].append(value)  # 累积到epoch
            self._step_buffer[name].append(value)   # 累积到step

    def compute_epoch_metrics(self):
        """计算并返回当前阶段的Epoch平均指标"""
        epoch_metrics = {}
        for name, values in self._epoch_buffer.items():
            avg_value = self._compute_avg(values)
            epoch_metrics[name] = avg_value
            self._history['epoch'][f"{self.current_stage}_{name}_epoch"].append(avg_value)
        self._epoch_buffer.clear()  # 清空Epoch缓冲区
        return epoch_metrics

    def compute_step_metrics(self):
        """计算并返回当前阶段的Step平均指标（自动清空Step缓冲区）"""
        step_metrics = {}
        for name, values in self._step_buffer.items():
            avg_value = self._compute_avg(values)
            step_metrics[name] = avg_value
            self._history['step'][f"{self.current_stage}_{name}_step"].append(avg_value)
        self._step_buffer.clear()  # 清空Step缓冲区
        return step_metrics

    def _compute_avg(self, values):
        """通用平均值计算（支持标量和张量）"""
        if not values:
            return 0.0  # 避免空列表
        if isinstance(values[0], (int, float)):
            return sum(values) / len(values)
        elif isinstance(values[0], torch.Tensor):
            return torch.stack(values).mean(dim=0)
        else:
            raise TypeError(f"Unsupported data type: {type(values[0])}")

    def get_history(self):
        """获取所有历史记录（用于可视化）"""
        return self._history
    


class Visualization:
    '''接受MetricTracker计算的_history，自动绘图。'''

    def refresh_plot(self, history: defaultdict[list]):
        '''再jupyter中持续刷新展示图片'''
        plt.close()                                 # close figure （推荐）
        fig = self._show(history)
        display.display(fig)                        # 在jupyter中展示 （推荐）
        display.clear_output(wait= True)             # 等待 （必须） 

    def _show(self, history: defaultdict[list]):
        '''根据实验：train、val、test等，指标：loss、acc、f1等自动绘图'''
        experiments, metrics = self._get_config(history)
        fig, axess = plt.subplots(nrows= len(history.keys()), ncols= len(metrics))        
        for i, strategy in enumerate(history.keys()):
            for j, metric in enumerate(metrics):
                for experiment in experiments:
                    axess[i][j].plot(history[strategy][f"{experiment}_{metric}_{strategy}"], label= f"{experiment}_{metric}")
                    axess[i][j].legend()
                    axess[i][j].set_xlabel(strategy)
                    axess[i][j].set_ylabel(metric)
                    axess[i][j].set_title(f"{metric} curve")
        fig.tight_layout()         
        return fig           

    def _get_config(self, history):
        '''获得实验：train、val、test等，指标：loss、acc、f1等'''
        experiments = set()
        metrics = set()

        for i in next(iter(history.values())).keys(): # 只去第一个值
            experiment_name, metrics_name, _ = i.split("_")
            experiments.add(experiment_name)
            metrics.add(metrics_name)
        return experiments, metrics
    

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    

class Callback:
    '''callback template'''

    def on_train_begin(self, **kwargs):
        pass
    
    def on_train_end(self, **kwargs):
        pass
    
    def on_epoch_begin(self, **kwargs):
        pass
    
    def on_epoch_end(self, **kwargs):
        pass
    
    def on_step_begin(self, **kwargs):
        pass
    
    def on_step_end(self, **kwargs):
        pass


class DemoCallback(Callback):
    def on_train_begin(self, **kwargs):
        print("Runing on_train_begin ...")


class ParameterSize:
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_parameter_size(self, model, dtype=torch.float32):
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        total_params = self.count_parameters(model)
        total_size = total_params * bytes_per_param
        print(f'{total_params/1000000} M parameters')
        print(f'{total_size/(1024*1024):.2f} MB')
        # return total_params, total_size


class Trainer:
    def __init__(
            self, 
            device = "auto",
            train_dataloader = None, 
            val_dataloader = None, 
            model = None, 
            loss_fn = None, 
            optimizer = None, 
            is_tqdm = True, 
            callbacks: list = [],
    ):
        # basic sets
        self.device = self._get_device(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = self._get_model(model)
        self.loss_fn = loss_fn 
        self.optimizer = optimizer
        self.is_tqdm = is_tqdm
        self.callbacks = callbacks

        # set metrics_tracker 
        self.metrics_tracker = MetricTracker()
        self.metrics_tracker.add_metric('loss', lambda **kw: kw['loss'])  # 直接从kwargs获取loss
        self.metrics_tracker.add_metric('acc', lambda **kw: (kw['y_hat'].argmax(1) == kw['y']).float().mean().item())

        # visualization
        self.visualization = Visualization()

    def _get_device(self, device):
        '''CPU or GPUs.'''
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = torch.device(device)
        print("=" * 100)
        print(f"Runing on {device} ...")
        print("=" * 100)
        return device
    
    def _get_model(self, model):
        '''Move the mode to device.'''
        model = torch.nn.DataParallel(model).to(self.device)
        return model
    
    def _disable_visualization(self):
        '''Weather show tqdm.'''
        if self.is_tqdm:
            return False 
        else:
            return True
        
    def _call_callbacks(self, method_name, **kwargs):
        '''Run the method of callback from callback dict with default order.'''
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                method = getattr(callback, method_name, f"{method_name} is not exist!")
                return method(**kwargs)
        
    def train(self, epochs: int, **kwargs):
        '''Main loop.'''
        self._call_callbacks(method_name= "on_train_begin", **kwargs)

        with tqdm(range(epochs), desc= "Training epoch", unit= "epoch", disable= self._disable_visualization()) as pbar:
            for epoch in pbar:
                # train
                self._call_callbacks(method_name= "on_epoch_begin", **kwargs)
                train_logs = self._train_step()
                train_logs = {"train_"+name: value for name, value in train_logs.items()}
                self._call_callbacks(method_name= "on_epoch_end", **kwargs)

                # val
                val_logs = self._validate_step()
                val_logs = {"val_"+name: value for name, value in val_logs.items()}    

                # update show progress bar or visualization
                pbar.set_postfix({**train_logs, **val_logs})
                if self._disable_visualization():
                    self.visualization.refresh_plot(history= self.metrics_tracker.get_history())

        self._call_callbacks(method_name= "on_train_end", **kwargs)

    def _train_step(self, **kwargs):
        '''On train step.'''
        self.model.train() 
        self.metrics_tracker.set_stage("train") ## for train

        for X, y in self.train_dataloader:
            self._call_callbacks(method_name= "on_step_begin", **kwargs)

            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(X)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()

            self.metrics_tracker.update(y_hat= y_hat, y= y, loss= loss.item()) ## update for train
            self.metrics_tracker.compute_step_metrics() ## on step level with train
            self._call_callbacks(method_name= "on_step_end", **kwargs)

        train_metrics = self.metrics_tracker.compute_epoch_metrics() ## on epoch level with train
        return train_metrics

    def _validate_step(self):
        '''On validate step.'''
        self.model.eval()
        self.metrics_tracker.set_stage("val") ## for val

        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)

                self.metrics_tracker.update(y_hat= y_hat, y= y, loss= loss.item()) ## update for val
                self.metrics_tracker.compute_step_metrics() ## on step level with val

            val_metrics = self.metrics_tracker.compute_epoch_metrics() ## on epoch level with val
        return val_metrics
    
    def save_metrics(self, file_path):
        '''Save the history with pickle format.'''
        history = self.metrics_tracker.get_history()
        with open(file_path, 'wb') as f:
            pickle.dump(history, f)

    def save_checkpoint(self, file_path):
        '''Save checkpoint.'''
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path):
        '''Load checkpoint.'''
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(state_dict= checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# trainer.metrics_tracker._history
# trainer.metrics_tracker.get_history()
# trainer.save_metrics(file_path= "./cache/metrics_tracker_history.pickle")
# trainer.save_checkpoint(file_path= './cache/checkpoint.pt')
# trainer.load_checkpoint(file_path= './cache/checkpoint.pt')
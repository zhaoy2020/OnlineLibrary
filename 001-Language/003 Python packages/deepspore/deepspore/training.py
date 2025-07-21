import random 
import pickle 
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from collections import defaultdict
from IPython import display
from tqdm import tqdm 
import time


def set_seed(seed: int = 42)-> None:
    '''
    Function for setting the seed.
    Args:
        seed: int, default is 42.

    Demo:
    >>>set_seed(seed= 123)
    '''

    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # GPU operation have separate seed
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set seed {seed} for reproducibility.")
    return None


def try_gpu(i:int=0):
    '''Try get gpu.
    >>>try_gpu(i=0)
    '''
    if torch.cuda.is_available():
        return torch.device(f"cuda:{i}")
    return torch.device('cpu')


def try_all_gpus():
    '''Try all GPUs.
    >>>try_all_gpus()
    '''
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


class Timer:
    """
    Record multiple running times.
    
    Demo:
    >>>timer = Timer()
    >>>timer.start()
    >>>timer.stop()
    >>>timer.sum()
    >>>timer.avg()
    >>>timer.cumsum()
    >>>timer.to_date(seconds= timer.sum())
    """

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

    def avg(self) -> float:
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self) -> float:
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self) -> list:
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    
    def to_date(self, seconds) -> None:
        '''Translate seconds to date format.'''
        days = seconds // (24 * 3600)
        hours = (seconds % (24 * 3600)) // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        print('='*20, '\n', f"Total：\n {days} d \n {hours} h \n {minutes} m \n {remaining_seconds} s")
        return None
        
    

class Callback:
    '''
    callback template
    '''

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
    '''
    Call function.
    '''
    def on_train_begin(self, **kwargs):
        '''Runing on train stage begin.'''
        print("Runing on_train_begin ...")


class GetModelSize:
    '''
    Calculate the parameter numbers and sizes of model.

    Demo:
    >>>get_model_size = GetModelSize()
    >>>get_model_size.parameter_numbers(model= model)
    >>>get_model_size.parameter_sizes(model= model)
    '''

    def parameter_numbers(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def parameter_sizes(self, model, dtype=torch.float32):
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        total_params = self.parameter_numbers(model)
        total_size = total_params * bytes_per_param
        print(f'{total_params/1000000} M parameters')
        print(f'{total_size/(1024*1024):.2f} MB')
        # return total_params, total_size


class Visualization:
    '''
    The Visualization class is designed for visualization of history.
    Demo:
    >>>visualization = Visualization()
    >>>visualization.refresh_plot(history=history)
    '''
    def _get_config(self, history):
        levels = set()
        stages = set()
        metrics = set()
        for level in history.keys():
            levels.add(level)
        for i in history.values():
            if i.__len__():
                for j in i.keys():
                    stage, metric = j.split("_")
                    stages.add(stage)
                    metrics.add(metric)
        return levels, stages, metrics
    
    def _show(self, history):
        '''根据实验：train、val、test等，指标：loss、acc、f1等自动绘图'''
        levels, stages, metrics = self._get_config(history)
        if len(metrics) == 0:
            print("⚠️ No metrics to plot. Skipping visualization.")
        fig, axess = plt.subplots(nrows= len(levels), ncols= len(metrics))        
        for i, level in enumerate(levels):
            for j, metric in enumerate(metrics):
                for stage in stages:
                    axess[i][j].plot(history[level][f"{stage}_{metric}"], label= f"{stage}_{metric}")
                    axess[i][j].legend()
                    axess[i][j].set_xlabel(level)
                    axess[i][j].set_ylabel(metric)
                    axess[i][j].set_title(f"{metric} curve")
        fig.tight_layout()         
        return fig
    
    def refresh_plot(self, history):
        '''再jupyter中持续刷新展示图片'''
        plt.close()                                 # close figure （推荐）
        fig = self._show(history)
        display.display(fig)                        # 在jupyter中展示 （推荐）
        display.clear_output(wait= True)             # 等待 （必须）


class MetricTracker:
    '''
    The MetricTracker class is designed to track and average metrics during training and validation.
    It allows for the addition of custom metrics, updates during training and validation steps,
    and provides methods to compute averages and store results in a history dictionary.
    
    Demo:
    >>>tracker = MetricTracker()
    >>>tracker.add_metric('loss', lambda **kw: kw['loss'])  # Example metric function
    >>>tracker.update(stage='val', y_hat=y_hat, y=y, loss=loss.item())
    >>>tracker.average_metrics(stage='val', show_level='step')
    >>>tracker.average_metrics(stage='val', show_level='epoch')
    >>>tracker.train_update(y_hat=y_hat, y=y, loss=loss.item())
    >>>tracker.train_average_metrics(level='step')
    >>>tracker.train_average_metrics(level='epoch')
    >>>tracker.get_history()
    >>>tracker.get_buffer()
    '''
    def __init__(self):
        '''
       _buffers = {
            'val': {
                'loss': [],
                'acc': [],
            },
            'test': {
                'loss': [],
                'acc': []
            }
        }
        _train_buffers = {
            'epoch': {
                'loss': [],
                'acc': [],
            },
            'step': {
                'loss': [],
                'acc': []
            }
        }
        history = {
            'epoch': {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
            },
            'step': {
                'train_loss': [],
                'trian_acc': [],
                'val_loss': [],
                'val_acc': [],
            },
        }
        '''
        self._metrics = {}
        # for val, test, infer just foucs on epoch level result.
        self._buffers = {}
        # for trian, has two models include epoch or step level.
        self._train_buffers = {
            'epoch': defaultdict(list),
            'step': defaultdict(list)
        }
        self.history = {
            'epoch': defaultdict(list),
            'step': defaultdict(list),
        }

    def add_metric(self, metric_name, metric_fn):
        '''Add metric.'''
        if metric_name not in self._metrics:
            self._metrics[metric_name] = metric_fn 

    def update(self, stage:str, **kwargs):
        '''Record metric results per batch step for val/test/infer and etc.'''
        # for val/test/infer which only foucs on epoch result.
        current_buffer = self._buffers.setdefault(stage, defaultdict(list))

        for metric_name, metric_fn in self._metrics.items():
            metric_result = metric_fn(**kwargs)
            current_buffer[metric_name].append(metric_result)

    def average_metrics(self, stage:str, show_level:str):
        '''Average the _buffer.'''
        current_buffer = self._buffers[stage]
        metric_results = {}
        for metric_name, values in current_buffer.items():
            value_avg = self._compute_avg(values)
            metric_results[metric_name] = value_avg
            # self.history[show_level][f'{stage}_{metric_name}_{show_level}'].append(value_avg)
            self.history[show_level][f'{stage}_{metric_name}'].append(value_avg)
        current_buffer.clear() # clear the buffer.
        return metric_results

    def train_update(self, **kwargs):
        for metric_name, metric_fn in self._metrics.items():
            metric_result = metric_fn(**kwargs)
            for level in ['epoch', 'step']: # 'epoch' or 'step'
                self._train_buffers[level][metric_name].append(metric_result)

    def train_average_metrics(self, level):
        '''Average.'''
        metric_results = {}
        for metric_name, values in self._train_buffers[level].items():
            value_avg = self._compute_avg(values)
            metric_results[metric_name] = value_avg 
            self.history[level][f'train_{metric_name}'].append(value_avg)
        self._train_buffers[level].clear() # clear the buffer.
        return metric_results 

    def _compute_avg(self, values):
        '''Compute the average.'''
        if not values:
            return 0.0  # 避免空列表
        if isinstance(values[0], (int, float)):
            return sum(values) / len(values)
        elif isinstance(values[0], torch.Tensor):
            return torch.stack(values).mean(dim=0)
        else:
            raise TypeError(f"Unsupported data type: {type(values[0])}")
        
    def get_history(self):
        '''Get the history.'''
        return self.history 
    
    def get_buffer(self):
        '''Get the buffer.'''
        return self._buffers


class Trainer:
    '''
    From Trainer, train visualization per step, valid visualization per epoch.
    Demo:
    >>>trainer = Trainer2()
    >>>trainer.train(epochs=3, steps=None)  # Visualization per epoch
    >>>trainer.train(epochs=3, steps=10)    # Visualization per 10 steps
    '''

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
        self.model = self._get_model(device, model)
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
    
    def _get_model(self, device, model):
        '''Move the mode to device.'''
        if device == 'auto':
            model = torch.nn.DataParallel(model).to(self.device)
        else:
            model = model.to(device)
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
    
    def _move_to_device(self, batch):
        '''Move value to device among different datatype (Dict, List, Tuple)'''
        if isinstance(batch, dict):
            # If batch is a dictionary, move each tensor to the device
            # This is common in NLP tasks where the batch is a dict of tensors
            # e.g., {'input_ids': tensor, 'attention_mask': tensor, 'labels': tensor}
            # Ensure that 'labels' is included in the batch
            if 'labels' not in batch:
                raise ValueError("Batch must contain 'labels' key for training.")
            return {key:value.to(self.device) for key, value in batch.items()}
        elif isinstance(batch, (list, tuple)):
            # If batch is a list or tuple, assume it's a tuple of tensors
            # e.g., (input_tensor, target_tensor)
            if len(batch) < 2:
                raise ValueError("Batch must contain at least two elements: input and labels.")
            return {'inputs': batch[0].to(self.device), 'labels': batch[-1].to(self.device)}
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}. Expected dict, list, or tuple.")

    def train(self, epochs: int=2, steps=None, **kwargs):
        '''Main loop.
        >>>train(epochs= 3, steps=None)
        >>>train(epochs= 3, steps=10)
        '''
        self._call_callbacks(method_name= "on_train_begin", **kwargs)

        with tqdm(range(epochs), desc= "Training epoch", unit= "epoch", disable= self._disable_visualization()) as pbar:
            steps_counter = 1
            for epoch in pbar:
                # train
                self._call_callbacks(method_name= "on_epoch_begin", **kwargs)
                steps_counter = self._train_step(steps, steps_counter, pbar)
                train_logs = self.metrics_tracker.train_average_metrics(level='epoch')
                train_logs = {"train_"+name: value for name, value in train_logs.items()}

                # update show progress bar or visualization
                if steps == None:
                    # val
                    self._validate_step()
                    val_logs = self.metrics_tracker.average_metrics(stage='val', show_level='epoch')
                    val_logs = {"val_"+name: value for name, value in val_logs.items()}

                    # visualization
                    ## on pbar
                    pbar.set_postfix({**train_logs, **val_logs})
                    ## on plot
                    if self._disable_visualization():
                        self.visualization.refresh_plot(history= self.metrics_tracker.get_history())

                self._call_callbacks(method_name= "on_epoch_end", **kwargs)

        self._call_callbacks(method_name= "on_train_end", **kwargs)

    def _train_step(self, steps, steps_counter, pbar, **kwargs):
        '''On train step.'''
        self.model.train() 

        for batch in self.train_dataloader:
            self._call_callbacks(method_name= "on_step_begin", **kwargs)

            batch = self._move_to_device(batch)
            y = batch['labels']
            self.optimizer.zero_grad()
            y_hat = self.model(**batch)
            loss = self.loss_fn(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度剪切
            self.optimizer.step()

            self.metrics_tracker.train_update(y_hat= y_hat, y= y, loss= loss.item()) ## update for train

            # update show progress bar or visualization
            if (steps != None) and (steps_counter%steps == 0):
                train_logs = self.metrics_tracker.train_average_metrics(level='step') ## on step level with train
                train_logs = {"train_"+name: value for name, value in train_logs.items()}
                # val
                self._validate_step()
                val_logs = self.metrics_tracker.average_metrics(stage='val', show_level='step')
                val_logs = {"val_"+name: value for name, value in val_logs.items()}
                pbar.set_postfix({**train_logs, **val_logs})
                if self._disable_visualization():
                    self.visualization.refresh_plot(history= self.metrics_tracker.get_history())

            self._call_callbacks(method_name= "on_step_end", **kwargs)
            steps_counter += 1
        return steps_counter
    
    def _validate_step(self):
        '''On validate step.'''
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                y = batch['labels']
                y_hat = self.model(**batch)
                loss = self.loss_fn(y_hat, y)
                self.metrics_tracker.update(stage='val', y_hat= y_hat, y= y, loss= loss.item())

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
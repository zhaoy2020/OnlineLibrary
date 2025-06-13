# This file is generated from saved notebook code blocks

#@save 
import json


def extract_save_blocks(ipynb_path:str, output_py_path:str):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    saved_code_blocks = []

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            if source_lines and source_lines[0].lstrip().startswith('#@save'):
                code_block = ''.join(source_lines)
                saved_code_blocks.append(code_block)

    if saved_code_blocks:
        with open(output_py_path, 'w', encoding='utf-8') as f_out:
            f_out.write("# This file is generated from saved notebook code blocks\n\n")
            f_out.write("\n\n".join(saved_code_blocks))
        print(f"Saved {len(saved_code_blocks)} block(s) to {output_py_path}")
    else:
        print("No #@save blocks found.")

#@save
import torch
from collections import defaultdict


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
    

#@save
import matplotlib.pyplot as plt 


def set_plt_default():
    plt.rcdefaults()
    

def set_plt_rcParams(**kswargs):
    # 设置字体栈（优先级从高到低）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Times New Roman',   # 英文优先使用
        'SimSun',            # 中文宋体
        # 'SimHei',            # 备用中文字体黑体
        # 'Noto Sans CJK SC'   # 最后回退
    ]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['pdf.fonttype'] = 42           # ai可编辑的字体格式
    plt.rcParams['figure.figsize'] = (3, 3)     # figsize
    plt.rcParams['savefig.format'] = "svg"      # svg格式
    plt.rcParams['savefig.transparent'] = True  # 背景是否透明

#@save
# %config InlineBackend.figure_format = 'svg'

from IPython import display
import matplotlib.pyplot as plt 
from collections import defaultdict


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
        for i in next(iter(history.values())).keys():           # 只取第一个值
            experiment_name, metrics_name, _ = i.split("_")
            experiments.add(experiment_name)
            metrics.add(metrics_name)
        return experiments, metrics

#@save
import time 


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

#@save
from abc import ABC 


class Callback(ABC):
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

#@save
import torch 
from tqdm import tqdm 
import pickle 


class Trainer:
    def __init__(
            self, 
            device: str = "auto",
            train_dataloader: torch.utils.data.DataLoader = None, 
            val_dataloader: torch.utils.data.DataLoader = None, 
            model: torch.nn.Module = None, 
            loss_fn:torch.nn.modules.loss = None, 
            optimizer: torch.optim.Optimizer = None, 
            is_tqdm: bool = True, 
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

    def _get_device(self, device: str) -> torch.device:
        '''CPU or GPUs.'''
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = torch.device(device)
        print("=" * 100)
        print(f"Runing on {device} ...")
        print("=" * 100)
        return device
    
    def _get_model(self, model) -> torch.nn.Module:
        '''Move the mode to device.'''
        model = torch.nn.DataParallel(model).to(self.device)
        return model
    
    def _disable_visualization(self) -> bool:
        '''Weather show tqdm.'''
        if self.is_tqdm:
            return False 
        else:
            return True
        
    def _call_callbacks(self, method_name: str, **kwargs):
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

    def _train_step(self, **kwargs) -> dict:
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

    def _validate_step(self) -> dict:
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
    
    def save_metrics(self, file_path: str):
        '''Save the history with pickle format.'''
        history = self.metrics_tracker.get_history()
        with open(file_path, 'wb') as f:
            pickle.dump(history, f)

    def save_checkpoint(self, file_path: str):
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

#@save
import torch 
from torch import nn


class Model(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, batch_first, num_layers):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first), 
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first), 
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.encoder(x) # 编码
        x = self.decoder(x) # 解码
        return x

#@save 
import torch 


class ParameterSize:
    def count_parameters(self, model: torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_parameter_size(self, model: torch.nn.Module, dtype= torch.float32):
        bytes_per_param = torch.tensor([], dtype=dtype).element_size()
        total_params = self.count_parameters(model)
        total_size = total_params * bytes_per_param
        parameter_number_M = total_params/1000000
        parameter_size_MB = total_size/(1024*1024)
        print(f'{parameter_number_M:.2f} M parameters')
        print(f'{parameter_size_MB:.2f} MB')
        return parameter_number_M, parameter_size_MB

#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

#@save
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


# 测试
batch_size, seq_len, embed_size = 2, 3, 4

# 实例化对象
add_norm = AddNorm(normalized_shape=embed_size, dropout=0.5)
add_norm.eval()

# 测试
X = torch.ones(size=(batch_size, seq_len, embed_size))

add_norm(X=X, Y=X).shape

#@save
class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        # X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        X = self.pos_encoding(self.embedding(X) * torch.sqrt(torch.tensor(self.num_hiddens)))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            # self.attention_weights[i] = blk.attention.attention.attention_weights_numpy
        return X


# 测试
encoder = TransformerEncoder(
    vocab_size=200, 
    key_size=24, 
    query_size=24, 
    value_size=24, 
    num_hiddens=24, 
    norm_shape=[100, 24], 
    ffn_num_input=24, 
    ffn_num_hiddens=48, 
    num_heads=8, 
    num_layers=2, 
    dropout=0.5
)
encoder.eval()

encoder(torch.ones((2, 100), dtype=torch.long)).shape
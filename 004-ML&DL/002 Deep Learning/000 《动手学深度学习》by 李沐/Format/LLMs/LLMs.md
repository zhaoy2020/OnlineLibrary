**Table of contents**<a id='toc0_'></a>    
- 1. [HuggingFace](#toc1_)    
- 2. [Transformer](#toc2_)    
- 3. [BERT](#toc3_)    
  - 3.1. [tokenizer](#toc3_1_)    
    - 3.1.1. [train a new tokenizer](#toc3_1_1_)    
    - 3.1.2. [make tokenizer to be used in transformers with AutoTokenizer](#toc3_1_2_)    
  - 3.2. [model](#toc3_2_)    
  - 3.3. [datas](#toc3_3_)    
  - 3.4. [trainer](#toc3_4_)    
- 4. [GPT](#toc4_)    
- 5. [T5](#toc5_)    
- 6. [BART](#toc6_)    
- 7. [LLaMa](#toc7_)    
  - 7.1. [Tokenizer](#toc7_1_)    
- 8. [DeepSeek](#toc8_)    
  - 8.1. [R1](#toc8_1_)    
- 9. [什么是RAG？](#toc9_)    
  - 9.1. [文本知识检索](#toc9_1_)    
    - 9.1.1. [知识库构建](#toc9_1_1_)    
    - 9.1.2. [查询构建](#toc9_1_2_)    
    - 9.1.3. [如何检索？-文本检索](#toc9_1_3_)    
    - 9.1.4. [如何喂给大模型？-生成增强](#toc9_1_4_)    
  - 9.2. [多模态知识检索](#toc9_2_)    
  - 9.3. [应用](#toc9_3_)    
- 10. [部署大模型](#toc10_)    
  - 10.1. [下载模型](#toc10_1_)    
  - 10.2. [ollama](#toc10_2_)    
    - 10.2.1. [Install and run model](#toc10_2_1_)    
    - 10.2.2. [API on web port](#toc10_2_2_)    
    - 10.2.3. [Python ollama module](#toc10_2_3_)    
      - 10.2.3.1. [demo：翻译中文为英文](#toc10_2_3_1_)    
  - 10.3. [ktransformers](#toc10_3_)    
    - 10.3.1. [Docker安装](#toc10_3_1_)    
    - 10.3.2. [编译安装](#toc10_3_2_)    
      - 10.3.2.1. [prepare](#toc10_3_2_1_)    
      - 10.3.2.2. [方式一：pip3 install whl](#toc10_3_2_2_)    
      - 10.3.2.3. [方式二：编译](#toc10_3_2_3_)    
    - 10.3.3. [虚拟机中编译安装](#toc10_3_3_)    
    - 10.3.4. [使用](#toc10_3_4_)    
- 11. [启动子预测](#toc11_)    
- 12. [转格式](#toc12_)    

<!-- vscode-jupyter-toc-config
	numbering=true
	anchor=true
	flat=false
	minLevel=1
	maxLevel=6
	/vscode-jupyter-toc-config -->
<!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

# 1. <a id='toc1_'></a>[HuggingFace](#toc0_)


```python
import os 


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```


```bash
%%bash
export HF_ENDPOINT="https://hf-mirror.com"
```

# 2. <a id='toc2_'></a>[Transformer](#toc0_)


```python

```

# 3. <a id='toc3_'></a>[BERT](#toc0_)

## 3.1. <a id='toc3_1_'></a>[tokenizer](#toc0_)

### 3.1.1. <a id='toc3_1_1_'></a>[train a new tokenizer](#toc0_)


```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, AutoModelForMaskedLM


# 初始化一个空的 WordPiece 模型
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# 设置训练参数
trainer = WordPieceTrainer(
    vocab_size=30000,        # 词汇表大小
    min_frequency=2,         # 最小词频
    show_progress=True,      # 显示进度
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)
```


```python
# 训练
tokenizer.train(files=["data/huggingface/dna_1g.txt"], trainer=trainer)

# 保存
tokenizer.save("data/huggingface/dna_wordpiece_dict.json")
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    Cell In[7], line 2
          1 # 训练
    ----> 2 tokenizer.train(files=["data/huggingface/dna_1g.txt"], trainer=trainer)
          4 # 保存
          5 tokenizer.save("data/huggingface/dna_wordpiece_dict.json")


    Exception: No such file or directory (os error 2)


### 3.1.2. <a id='toc3_1_2_'></a>[make tokenizer to be used in transformers with AutoTokenizer](#toc0_)


```python
new_tokenizer = Tokenizer.from_file("data/huggingface/dna_wordpiece_dict.json")

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=new_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 保存
wrapped_tokenizer.save_pretrained("data/huggingface/dna_wordpiece_dict")
```




    ('data/huggingface/dna_wordpiece_dict/tokenizer_config.json',
     'data/huggingface/dna_wordpiece_dict/special_tokens_map.json',
     'data/huggingface/dna_wordpiece_dict/tokenizer.json')




```python
from transformers import AutoTokenizer


# 加载
tokenizer = AutoTokenizer.from_pretrained("data/huggingface/dna_wordpiece_dict")
#tokenizer.pad_token = tokenizer.eos_token
```


```python
# 编码
tokenizer("ATCGGATCG")
```




    {'input_ids': [6, 766, 22, 10], 'token_type_ids': [0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1]}




```python
tokenizer
```




    PreTrainedTokenizerFast(name_or_path='data/huggingface/dna_wordpiece_dict', vocab_size=30000, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
    	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	1: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	2: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	3: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    	4: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    }



## 3.2. <a id='toc3_2_'></a>[model](#toc0_)


```python
from transformers import BertConfig, BertForMaskedLM 


# 配置
max_len = 1024 

config = BertConfig(
    vocab_size = len(tokenizer),
    max_position_embeddings=max_len, 
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
) 

# 模型
model = BertForMaskedLM(config=config)
```

    BertForMaskedLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
      - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
      - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
      - If you are not the owner of the model architecture class, please contact the model code owner to update it.


## 3.3. <a id='toc3_3_'></a>[datas](#toc0_)


```python
from datasets import load_dataset 


raw_dataset = load_dataset('text', data_files='data/huggingface/dna_1g.txt')
raw_dataset
```




    DatasetDict({
        train: Dataset({
            features: ['text'],
            num_rows: 1079595
        })
    })




```python
dataset = raw_dataset["train"].train_test_split(test_size=0.1, shuffle=True)
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['text'],
            num_rows: 971635
        })
        test: Dataset({
            features: ['text'],
            num_rows: 107960
        })
    })




```python
tokenizer._tokenizer.model.max_input_chars_per_word = 10000


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_len)


# 对数据集应用分词函数
tokenized_datasets = dataset.map(tokenize_function, batched=False, remove_columns=['text'], num_proc=50)  # 设置为你的 CPU 核心数或根据需要调整

```


    Map (num_proc=50):   0%|          | 0/971635 [00:00<?, ? examples/s]



    Map (num_proc=50):   0%|          | 0/107960 [00:00<?, ? examples/s]



```python
from transformers import DataCollatorForLanguageModeling


# 创建一个数据收集器，用于动态填充和遮蔽,注意mlm=true
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
```


```python
dataset["train"][0]
```




    {'text': 'GAATATTTGTCTATTCTTCTTAACTTTCTCCACTGTAAATTAAATTGCTCCTCAGGGTGCTATATGGCATCCCTTGCTATTTTTGGAGCAAATCTTAAATTCTTCAACAATTTTATCAAGACAAACACAACTTTCAGTAAATTCATTGTTTAAATTTGGTGAAAAGTCAGATTTCTTTACACATAGTAAAGCAAATGTAAAATAATATATCAATGTGATTCTTTTAATAAAATACCATTATTGCCAATGGTTTTTAATAGTTCACTGTTTGAAAGAGACCACAAAATTCATGTGCAAAAATCACAAGCATTCTTATACAACAGTGACAGACAAACAGAGAGCCAAATCAGGAATGAACTTCCATTCACAATTGCTTCAAAGAGAATCAAATACCTAGGAATCCAACTTACAAGGGATGTAAAGGACCTCTTCAAGGAGAACTACAAACCACTGCTCAGTGAAATAAAAGAGGACACAAACAAATGGAAGAACATACCATGCTCATGGATAGGAAGAATCAATATCGTGAAAATGGCCATACTGCCCAAGGTAATTTATAGATTCAATGCCATCCCCATCAAGCTACCAATGAGTTTCTTCACAGAATTGGAAAAAACTGTTTTAAAGTTCATATGGAACCAAAAAAGAACCCACATTGCCAAGACAATCCTAAGTCAAATGAACAAAGCTGGAGGGATCATGCTACCTGACTTCAAACTATACTACAAGGCTACAGTAACCAAAATAGCATGGTACTGGTACCAAAACAGAAATATAGACCAATGGAACAGCATAGAGTCCTCAGAAATAATACCACACATCTACATCTTTGATAAATCTGACAAAAACAAGAAATGGGGAAAGGATTCTCTATATAATAAATGGTGCTGGGAAAATTGGCTAGCCATAAGTAGAAAGCTGAAACTGGATCCTTTCCTTACTCTTTATACGAAAATTAATTCAAGATGGAGTAGAGACTTAAATGTTAGACCTAATACCA'}




```python
tokenizer.tokenize(dataset["train"][0]["text"][:100])
```




    ['GAA',
     '##TATTTG',
     '##TCTATT',
     '##CTTCTTAA',
     '##CTTTCTCC',
     '##A',
     '##CTGTAAATT',
     '##AAATT',
     '##GCTCC',
     '##TCAGG',
     '##GTGCTA',
     '##TATGGCA',
     '##TCCCTT',
     '##GCTATTTT',
     '##TGGAGCAA',
     '##A',
     '##TCTTAAA',
     '##T']



## 3.4. <a id='toc3_4_'></a>[trainer](#toc0_)


```python
from transformers import TrainingArguments, Trainer


run_path = "cache/bert_run"
train_epoches = 5
batch_size = 2


training_args = TrainingArguments(
        output_dir=run_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True, #v100没法用
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)
```

    Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.



```python
trainer.train()
trainer.save_model("cache/dna_bert_v0")
```

# 4. <a id='toc4_'></a>[GPT](#toc0_)


```python

```

# 5. <a id='toc5_'></a>[T5](#toc0_)


```python

```

# 6. <a id='toc6_'></a>[BART](#toc0_)


```python

```

# 7. <a id='toc7_'></a>[LLaMa](#toc0_)

## 7.1. <a id='toc7_1_'></a>[Tokenizer](#toc0_)


```python
import sentencepiece as spm


spm.SentencePieceTrainer.train(
    input="data/huggingface/dna_1g.txt,data/huggingface/protein_1g.txt", 
    model_prefix="dna_llama", 
    vocab_size=60000, 
    model_type="bpe", 
    # max_sentence_length=1000000,
    num_threads=50, 
)
```


```python
tokenizer = spm.SentencePieceProcessor(model_file="dna_llama.model")

tokenizer.encode("ATCGGATCG")

```

# 8. <a id='toc8_'></a>[DeepSeek](#toc0_)

## 8.1. <a id='toc8_1_'></a>[R1](#toc0_)


```python
# Use a pipeline as a high-level helper
from transformers import pipeline


messages = [
    {"role": "user", "content": "Who are you?"},
]

pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True)

pipe(messages)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[5], line 9
          2 from transformers import pipeline
          5 messages = [
          6     {"role": "user", "content": "Who are you?"},
          7 ]
    ----> 9 pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True)
         11 pipe(messages)


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/transformers/pipelines/__init__.py:895, in pipeline(task, model, config, tokenizer, feature_extractor, image_processor, framework, revision, use_fast, token, device, device_map, torch_dtype, trust_remote_code, model_kwargs, pipeline_class, **kwargs)
        893 if isinstance(model, str) or framework is None:
        894     model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]}
    --> 895     framework, model = infer_framework_load_model(
        896         model,
        897         model_classes=model_classes,
        898         config=config,
        899         framework=framework,
        900         task=task,
        901         **hub_kwargs,
        902         **model_kwargs,
        903     )
        905 model_config = model.config
        906 hub_kwargs["_commit_hash"] = model.config._commit_hash


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/transformers/pipelines/base.py:296, in infer_framework_load_model(model, config, model_classes, task, framework, **model_kwargs)
        294         for class_name, trace in all_traceback.items():
        295             error += f"while loading with {class_name}, an error is thrown:\n{trace}\n"
    --> 296         raise ValueError(
        297             f"Could not load model {model} with any of the following classes: {class_tuple}. See the original errors:\n\n{error}\n"
        298         )
        300 if framework is None:
        301     framework = infer_framework(model.__class__)


    ValueError: Could not load model deepseek-ai/DeepSeek-R1 with any of the following classes: (<class 'transformers.models.auto.modeling_auto.AutoModelForCausalLM'>,). See the original errors:
    
    while loading with AutoModelForCausalLM, an error is thrown:
    Traceback (most recent call last):
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 789, in urlopen
        response = self._make_request(
                   ^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 490, in _make_request
        raise new_e
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 466, in _make_request
        self._validate_conn(conn)
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 1095, in _validate_conn
        conn.connect()
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connection.py", line 652, in connect
        sock_and_verified = _ssl_wrap_socket_and_match_hostname(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connection.py", line 805, in _ssl_wrap_socket_and_match_hostname
        ssl_sock = ssl_wrap_socket(
                   ^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/util/ssl_.py", line 465, in ssl_wrap_socket
        ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/util/ssl_.py", line 509, in _ssl_wrap_socket_impl
        return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/ssl.py", line 455, in wrap_socket
        return self.sslsocket_class._create(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/ssl.py", line 1042, in _create
        self.do_handshake()
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/ssl.py", line 1320, in do_handshake
        self._sslobj.do_handshake()
    ConnectionResetError: [Errno 104] Connection reset by peer
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/requests/adapters.py", line 667, in send
        resp = conn.urlopen(
               ^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 843, in urlopen
        retries = retries.increment(
                  ^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/util/retry.py", line 474, in increment
        raise reraise(type(error), error, _stacktrace)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/util/util.py", line 38, in reraise
        raise value.with_traceback(tb)
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 789, in urlopen
        response = self._make_request(
                   ^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 490, in _make_request
        raise new_e
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 466, in _make_request
        self._validate_conn(conn)
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connectionpool.py", line 1095, in _validate_conn
        conn.connect()
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connection.py", line 652, in connect
        sock_and_verified = _ssl_wrap_socket_and_match_hostname(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/connection.py", line 805, in _ssl_wrap_socket_and_match_hostname
        ssl_sock = ssl_wrap_socket(
                   ^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/util/ssl_.py", line 465, in ssl_wrap_socket
        ssl_sock = _ssl_wrap_socket_impl(sock, context, tls_in_tls, server_hostname)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/urllib3/util/ssl_.py", line 509, in _ssl_wrap_socket_impl
        return ssl_context.wrap_socket(sock, server_hostname=server_hostname)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/ssl.py", line 455, in wrap_socket
        return self.sslsocket_class._create(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/ssl.py", line 1042, in _create
        self.do_handshake()
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/ssl.py", line 1320, in do_handshake
        self._sslobj.do_handshake()
    urllib3.exceptions.ProtocolError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/transformers/pipelines/base.py", line 283, in infer_framework_load_model
        model = model_class.from_pretrained(model, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 559, in from_pretrained
        return model_class.from_pretrained(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/transformers/modeling_utils.py", line 3589, in from_pretrained
        if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/transformers/utils/hub.py", line 655, in has_file
        response = get_session().head(
                   ^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/requests/sessions.py", line 624, in head
        return self.request("HEAD", url, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/requests/sessions.py", line 589, in request
        resp = self.send(prep, **send_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/requests/sessions.py", line 703, in send
        r = adapter.send(request, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 93, in send
        return super().send(request, *args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/bmp/backup/zhaosy/miniconda3/envs/pytorch/lib/python3.12/site-packages/requests/adapters.py", line 682, in send
        raise ConnectionError(err, request=request)
    requests.exceptions.ConnectionError: (ProtocolError('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer')), '(Request ID: 787d5459-e5a7-4279-8e49-e516f6e6aa22)')
    
    




```python
# Load model directly
from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
```

# 9. <a id='toc9_'></a>[什么是RAG？](#toc0_)

RAG的分类：

|Model | 检索器微调 | 大预言模型微调| 例如 |
|---|---|---| --- |
| 黑盒 | - | - | e.g. In-context ralm |
| 黑盒 | 是 | - | e.g. Rplug |
| 白盒 | - | 是 | e.g. realm, self-rag |
| 白盒 | 是 | 是 | e.g. altas |

## 9.1. <a id='toc9_1_'></a>[文本知识检索](#toc0_)
如何检索出相关信息来辅助改善大语言模型生成质量的系统。知识检索通常包括知识库构建、查询构建、文本检索和检索结果重排四部分。

### 9.1.1. <a id='toc9_1_1_'></a>[知识库构建](#toc0_)
文本块的知识库构建，如维基百科、新闻、论文等。

文本分块：将文本分成多个块，每个块包含一个或多个句子。
- 固定大小块：将文本分成固定大小的块，如每个块包含512个字符。
- 基于内容块：将文本分成基于内容的块，如每个块包含一个句子。
  - 通过句子分割符分割句子。
  - 用LLM进行分割

知识库增强：知识库增强是通过改进和丰富知识库的内容和结构，为查询提供"抓手”，包括查询生成与标题生成两种方法。
- 伪查询生成
- 标题生成

### 9.1.2. <a id='toc9_1_2_'></a>[查询构建](#toc0_)
查询构建：旨在通过查询增强的方式，扩展和丰富用户查询的语义和内容，提高检索结果的准确性和全面性，“钩"出相应内容。增强方式可分为语义增强与内容增强。
- 语义增强：同一句话多种表达方式
- 内容增强：增加背景知识

### 9.1.3. <a id='toc9_1_3_'></a>[如何检索？-文本检索](#toc0_)
`检索器`：给定知识库和用户查询，文本检索旨在找到知识库中与用户查询相关的知识文本;检索效率增强旨在解决检索时的性能瓶颈问题。所以检索质量、检索效率很重要。常见检索器有三类：
- 判别式检索器：
  - 稀疏检索器，e.g. TF-IDF
  - 双向编码检索器，e.g. 用bert预先将文本块进行编码成向量
  - 交叉编码检索器，e.g. 
- 生成式检索器：器直接将知识库中的文档信息记忆在模型参数中。然后，在接收到查询请求时，能够直接生成相关文档的标识符夺（即Doc ID），以完成检索。
- 图检索器：图检索器的知识库为图数据库，包括开放知识图谱和自建图两种，它们一般由<主体、谓词和客体>三元组构成。这样做不仅可以捕捉概念间的语义关系，还允许人类和机器可以共同对知识进行理解与推理。

`重排器`：检索阶段为了保证检索速度通常会损失一定的性能，可能检索到质量较低的文档。重排的目的是对检索到的段落进行进一步的排序精选。重排可以分为基于交叉编码的方法和基于上下文学习的方法。

### 9.1.4. <a id='toc9_1_4_'></a>[如何喂给大模型？-生成增强](#toc0_)
RAG增强比较：

|架构分类|优点|缺点|
|-|-|-|
|输入端prompt|简单|tokens太多|
|中间层|高效|耗GPU资源|
|输出端|-|-|

## 9.2. <a id='toc9_2_'></a>[多模态知识检索](#toc0_)
## 9.3. <a id='toc9_3_'></a>[应用](#toc0_)
对话机器人、知识库文答...

# 10. <a id='toc10_'></a>[部署大模型](#toc0_)


## 10.1. <a id='toc10_1_'></a>[下载模型](#toc0_)


```python
# https://huggingface.co/deepseek-ai/DeepSeek-R1
hfd.sh deepseek-ai/DeepSeek-R1 -x 10 -j 10 

# https://huggingface.co/unsloth/DeepSeek-R1-GGUF
hfd.sh unsloth/DeepSeek-R1-GGUF -x 10 -j 10 --include DeepSeek-R1-Q8_0

```

## 10.2. <a id='toc10_2_'></a>[ollama](#toc0_)
### 10.2.1. <a id='toc10_2_1_'></a>[Install and run model](#toc0_)


```python
# start the serve
ollama serve

# list all model images
ollama list 

# run model from image
ollama run model_card
```

### 10.2.2. <a id='toc10_2_2_'></a>[API on web port](#toc0_)
communicatation with local model via web port.

`generate` and `chat`.


```bash
%%bash
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-r1:7b",
  "prompt": "Who are you?",
  "stream": false,
  "options": {
    "temperature": 0.6
  },
  "format": "json"
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   456  100   315  100   141    142     64  0:00:02  0:00:02 --:--:--   206


    {"model":"deepseek-r1:7b","created_at":"2025-02-18T02:49:32.744934023Z","response":"{\"}\u003cthink\u003e{\"\n\n\n\n\n\n\n\n\n\n:\n\n{\n\n}\n\n}\n\n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n \n\n","done":false}


```bash
%%bash
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:7b",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ],
  "stream": false
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  5034    0  4905  100   129    326      8  0:00:16  0:00:15  0:00:01   719


    {"model":"deepseek-r1:7b","created_at":"2025-02-18T02:49:26.56791501Z","message":{"role":"assistant","content":"\u003cthink\u003e\nOkay, so I just read that \"Why is the sky blue?\" and now I'm trying to figure it out myself. Let me think through this step by step.\n\nFirst off, when you look at the sky on a clear day, it's usually blue, especially during the day when the sun is out. But sometimes I've seen it turn other colors too, like red in the evening or during sunrise. So why is it mostly blue?\n\nI know that light travels through the atmosphere, but how does it get colored? I remember learning about something called Rayleigh scattering from my science class. Let me try to recall what that was about. Rayleigh scattering involves light interacting with particles much smaller than the wavelength of light itself. When sunlight enters the Earth's atmosphere, it reaches tiny molecules in the air, like nitrogen and oxygen.\n\nWait, so these small particles scatter the sunlight in all directions. But why does this result in a blue sky? I think it has something to do with the wavelengths of light. Visible light ranges from violet to red, right? And I remember that violet light has a shorter wavelength than blue. So maybe the shorter wavelengths are scattered more.\n\nBut if blue is scattered more, wouldn't it be easier to see at night when there's less atmosphere overhead? Hmm, but then why does the sky turn red in the evening or during sunrise?\n\nOh right! During sunrise and sunset, the light has to pass through a much thicker layer of atmosphere. That makes sense because we're looking at the light after it's been scattered through more particles. The longer path means that all the shorter wavelengths (like violet) are scattered out, leaving red to dominate because it has a longer wavelength.\n\nSo during the day, when the sun is directly overhead, blue and green wavelengths get scattered away by Rayleigh scattering, making the sky appear blue. But in the early morning or late afternoon, as the sun is near the horizon, the light has to pass through more atmosphere, so red comes through because it's not scattered as much.\n\nWait, but isn't there also something called Mie scattering? I think that happens when particles are larger than the wavelength of light. Does that affect the color of the sky too?\n\nI believe Mie scattering is more significant for larger particles, like dust or droplets in clouds. So it might cause some effects we see during sunrise and sunset, but not as much as Rayleigh does for small molecules.\n\nSo to summarize: The sky appears blue on a clear day because blue light scatters more in the atmosphere due to Rayleigh scattering by tiny gas molecules. During sunrise and sunset, the longer path allows red light to dominate, giving the sky its reddish hues.\n\nBut wait, what about when we see other colors? I mean, sometimes during the day it's not just blue; I've seen green or yellow in some places. Is that because of Rayleigh scattering changing as the atmosphere gets thicker?\n\nOr maybe it's due to other factors like pollution or particles in the air affecting light differently. That might complicate things.\n\nAlso, does humidity play a role? Sometimes when it's humid, does the sky appear clearer instead of blue? I think higher humidity can affect how light scatters because more water vapor means more molecules to scatter off.\n\nSo maybe on days with high humidity, the Rayleigh scattering is less effective, making the sky look different. But that's probably a secondary effect compared to the basic reason for the color being blue.\n\nIn any case, I think the primary reason is Rayleigh scattering by nitrogen and oxygen in the atmosphere causing shorter wavelengths like blue to scatter more, resulting in a blue sky during clear days.\n\u003c/think\u003e\n\nThe sky appears blue primarily due to a phenomenon known as Rayleigh scattering. When sunlight enters Earth's atmosphere, it interacts with tiny molecules of nitrogen and oxygen. These particles scatter shorter wavelengths of light, such as blue and violet, which have shorter wavelengths than red or orange. This scattering is more effective for shorter wavelengths, causing the sky to appear blue during the day.\n\nDuring sunrise and sunset, the light passes through a thicker layer of atmosphere, where all shorter wavelengths are scattered away, leaving longer wavelengths like red to dominate, resulting in the reddish hues observed at these times.\n\nOther factors such as humidity, pollution, or atmospheric particles can influence how light scatters, but the primary reason for the sky's blue color remains Rayleigh scattering by nitrogen and oxygen molecules."},"done_reason":"stop","done":true,"total_duration":15024265818,"load_duration":22057058,"prompt_eval_count":9,"prompt_eval_duration":7000000,"eval_count":901,"eval_duration":14993000000}

### 10.2.3. <a id='toc10_2_3_'></a>[Python ollama module](#toc0_)


```python
import ollama


texts = '''
详细比较deepseek母公司和openAI公司的区别
'''

# model_card = "deepseek-r1:7b"
model_card = "modelscope.cn/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"

# 方式一（非流式输出）：
# outputs = ollama.generate(model_card, inputs)
# print(f'{outputs['response']}')

# 方式二（流式输出）：
outputs = ollama.generate(
    stream=True,
    model=model_card,
    prompt=texts,
)
for chunk in outputs:
    if not chunk['done']:
        print(f'{chunk['response']}', end='', flush=True)
```

#### 10.2.3.1. <a id='toc10_2_3_1_'></a>[demo：翻译中文为英文](#toc0_)


```python
import ollama 


class zh2en():
    def __init__(self, model_card):
        self.model_card = model_card
        
    def build_prompt(self, texts):
        # with open(prompt_template_path, 'r') as f:
        #     prompt_template = f.read()
        #     # str with replace function
        #     prompt = prompt_template.replace(var, texts)
        prompt_template = """
        专业翻译：\n
        ---\n
        {Chinese_words} \n
        --- \n
        作为翻译专家，将上述中文准确翻译为英文。 \n
        """
        prompt = prompt_template.replace("{Chinese_words}", texts)
        return prompt

    def translate(self, texts):
        prompt = self.build_prompt(texts = texts)
        # key step
        outputs = ollama.generate(
            stream=True,
            model=self.model_card,
            prompt=prompt,
        )
        for chunk in outputs:
            if not chunk['done']:
                print(f'{chunk['response']}', end='', flush=True)
            else:
                print('⚡')


translater = zh2en(model_card='modelscope.cn/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF')

translater.translate('基于深度学习的对枯草芽胞杆菌芽胞形成相关基因的研究。')
translater.translate('通过宏基因组研究微生物与植物相互作用的机制。')
```

    <think>
    嗯，首先我要理解这个题目的意思。“基于深度学习”指的是使用深度学习技术来进行研究。而“对枯草芽胞杆菌芽胞形成相关基因的研究”则是具体的研究内容，涉及到枯草芽胞杆菌在形成芽胞过程中相关的基因。
    
    我需要把这整个句子准确地翻译成英文。首先，“基于深度学习”可以直接翻译为“Based on deep learning”。接下来是“研究”，对应的英文是“study”。然后是“枯草芽胞杆菌”，这个应该是一个专有名词，可能需要查一下正确的英译名称，比如“Bacillus subtilis”。
    
    接着是“芽胞形成相关基因”，这部分可以翻译为“genes related to spore formation”。最后，把整个句子连贯起来，就是“Based on deep learning study of genes related to spore formation in Bacillus subtilis.”
    
    这样组合起来，既准确传达了原意，又符合英文的表达习惯。我觉得这个翻译应该是比较专业和准确的。
    </think>
    
    Study of Genes Related to Spore Formation in *Bacillus subtilis* Based on Deep Learning⚡
    <think>
    好的，首先我要理解用户的需求。他给了一个中英对照的句子，要求专业翻译，并且需要将中文句子“通过宏基因组研究微生物与植物相互作用的机制。”准确地翻译成英文。
    
    接下来，我需要分析原文的意思。句子的主干是“通过宏基因组研究…”，这里的关键词有“宏基因组”、“微生物”、“植物”以及“相互作用的机制”。所以，首先要确定这些术语在英文中的准确对应词。
    
    “宏基因组”通常翻译为“metagenome”或者“meta-genomics”，但更常见的是使用“metagenomics”来表示这一研究领域。因此，这里选择“metagenomics”作为翻译。
    
    然后，“通过…研究…”的结构在英文中可以用“through”或者“by means of”来表达，但为了简洁和专业，直接使用“Through”比较合适。
    
    接下来是“微生物与植物相互作用的机制”。这里需要注意语序和用词。整体结构应该是“the mechanisms underlying the interactions between microorganisms and plants.” 这样不仅清晰，而且符合学术写作的规范。
    
    最后，把这些部分组合起来，确保句子通顺且准确。所以，最终翻译为：“Through metagenomics research on the mechanisms of interaction between microorganisms and plants.”
    
    在整个思考过程中，我注意到用户可能是在撰写学术论文或者准备研究报告，因此准确性和专业性是关键。此外，保持句子的简洁也是必要的，以便读者能够快速理解内容。
    
    最后，再检查一遍翻译是否忠实于原文，并且符合英语表达习惯。确认无误后，就可以将这个翻译结果提供给用户了。
    </think>
    
    Through metagenomics research on the mechanisms of interaction between microorganisms and plants.⚡


## 10.3. <a id='toc10_3_'></a>[ktransformers](#toc0_)


### 10.3.1. <a id='toc10_3_1_'></a>[Docker安装](#toc0_)

[https://github.com/kvcache-ai/ktransformers-private/blob/main/doc/en/Docker.md](https://github.com/kvcache-ai/ktransformers-private/blob/main/doc/en/Docker.md)


```python
# pull the image from docker hub 
# about 19 GB
# docker pull approachingai/ktransformers:0.1.1
docker pull approachingai/ktransformers:0.2.1

# docker run \
#     --gpus all \
#     -v /path/to/models:/models \
#     -p 10002:10002 \
#     approachingai/ktransformers:v0.1.1 \
#     --port 10002 \
#     --gguf_path /models/path/to/gguf_path \
#     --model_path /models/path/to/model_path \
#     --web True

# Directly run
docker run  \
    -v /bmp/backup/zhaosy/ProgramFiles/hf/deepseek-ai:/models \
    -p 10002:10002 \
    approachingai/ktransformers:0.1.1 \
    --port 10002 \
    --model_path /bmp/backup/zhaosy/ProgramFiles/hf/deepseek-ai/DeepSeek-R1 \
    --gguf_path /bmp/backup/zhaosy/ProgramFiles/hf/deepseek-ai/DeepSeek-R1-Q4_K_M_GGUF \
    --web True

# or
docker run \
    # -d \
    --gpus all \
    -it \
    -p 10002:10002 \
    approachingai/ktransformers:0.1.1 \
    /bin/bash 

## and then
docker exec -it container_ID /bin/bash
```

QA:

- Q: Docker运行后出现 Illegal instruction (core dumped)报错
  - [https://github.com/kvcache-ai/ktransformers/issues/356](https://github.com/kvcache-ai/ktransformers/issues/356)
  - 重新编译以下:
    ```bash
    USE_NUMA=1
    bash install.sh
    ```

### 10.3.2. <a id='toc10_3_2_'></a>[编译安装](#toc0_)

[https://kvcache-ai.github.io/ktransformers/en/install.html](https://kvcache-ai.github.io/ktransformers/en/install.html)

#### 10.3.2.1. <a id='toc10_3_2_1_'></a>[prepare](#toc0_)


```python
name="ktransformers"

conda create -n $name python=3.11 -y 
conda activate $name


# Install CudaToolkit and nvcc ...
conda install nvidia/label/cuda-12.4.0::cuda -y --channel nvidia/label/cuda-12.4.0

# Anaconda provides a package called `libstdcxx-ng` that includes a newer version of `libstdc++`, which can be installed via `conda-forge`.
conda install -c conda-forge libstdcxx-ng -y 

strings ~/miniconda3/envs/${name}/lib/libstdc++.so.6 | grep GLIBCXX


# Install PyTorch via pip ...
# pip3 install torch torchvision torchaudio
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch -y

# pip3 install packaging ninja cpufeature numpy
conda install conda-forge::ninja conda-forge::packaging anaconda::numpy -y


```

#### 10.3.2.2. <a id='toc10_3_2_2_'></a>[方式一：pip3 install whl](#toc0_)


```python
# ktransformers
pip3 install ktransformers-0.2.1.post1+cu124torch24avx2-cp311-cp311-linux_x86_64.whl

# flash_attn
# pip3 install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# flashinfer
pip3 install flashinfer_python-0.2.2+cu124torch2.4-cp38-abi3-linux_x86_64.whl
python -c "import torch; print(torch.cuda.get_device_capability())"
export TORCH_CUDA_ARCH_LIST="8.0"
```

#### 10.3.2.3. <a id='toc10_3_2_3_'></a>[方式二：编译](#toc0_)


```python
# Make sure your system has dual sockets and double size RAM than the model's size (e.g. 1T RAM for 512G model)
export USE_NUMA=1
bash install.sh # or `make dev_install`
```

### 10.3.3. <a id='toc10_3_3_'></a>[虚拟机中编译安装](#toc0_)

由于`GLIBCxxx`报错，改用虚拟机中ubuntu安装ktransformers


```python
# 拉取包含GLIBC 2.29+的镜像（如Ubuntu 20.04）
docker pull ubuntu:20.04

# 启动容器并挂载项目目录 
# docker run -it -v /path/to/your/code:/app ubuntu:20.04 /bin/bash 
docker run \
    --gpus all \
    -p 10002:10002\
    -it \
    ubuntu:20.04 /bin/bash 

# Install wget and then 
apt-get update 
apt-get install wget 

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh

docker exec --gpus all -it ubuntu:20.04 /bin/bash
```

### 10.3.4. <a id='toc10_3_4_'></a>[使用](#toc0_)


```python
# 将主机和容器的10002端口做映射
docker run --gpus all -p 10002:10002 -it \
    ubuntu_2004:ktransformers /bin/bash


# 检查映射
docker port [容器ID]
```


```python
# As we use dual socket, we set cpu_infer to 65
python -m ktransformers.local_chat \
    --model_path DeepSeek-R1 \
    --gguf_path DeepSeek-R1-Q4_K_M_GGUF \
    --cpu_infer 60 \
    --max_new_tokens 10000 \
    --cache_lens 50000 \
    --total_context 50000 \
    --cache_q4 true \
    --temperature 0.6 \
    --top_p 0.95 \
    --force_think \
    --use_cuda_graph \
    --port 10002
    # --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml \
    # --host 127.0.0.1 \
```


```python
# http://localhost:10002/web/index.html#/chat
ktransformers \
    --model_path DeepSeek-R1/ \
    --gguf_path DeepSeek-R1-Q4_K_M_GGUF/ \
    --port 10002 \
    --web True
```


```bash
%%bash
curl -X 'GET' \
  'http://localhost:10002/api/tags' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "tell a joke",
      "role": "user"
    }
  ],
  "model": "DeepSeek-R1",
  "stream": true
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   213  100    81  100   132  40500  66000 --:--:-- --:--:-- --:--:--  104k


    {"models":[{"name":"DeepSeek-Coder-V2-Instruct","modified_at":"123","size":123}]}


```bash
%%bash
curl -X 'POST' \
  'http://localhost:10002/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "tell a joke",
      "role": "user"
    }
  ],
  "model": "DeepSeek-R1",
  "stream": true
}'
```


```bash
%%bash
curl -X 'POST' \
  'http://localhost:10002/api/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  "model": "DeepSeek-R1",
  "prompt": "tell me a joke",
  "stream": true
}"
```

# 11. <a id='toc11_'></a>[启动子预测](#toc0_)


```python

```

# 12. <a id='toc12_'></a>[转格式](#toc0_)


```bash
%%bash
# ipynb to html
jupyter nbconvert \
    --to html LLMs.ipynb \
    --output-dir=./Format/LLMs \
    # --NbConvertApp.log_level=ERROR

cp -rf Pytorch_Pictures ./Format/LLMs/
# browse translate html to pdf
```

    [NbConvertApp] Converting notebook LLMs.ipynb to html
    [NbConvertApp] Writing 405891 bytes to Format/LLMs/LLMs.html



```python
# ipynb to markdown
!jupyter nbconvert --to markdown LLMs.ipynb --output-dir=./Format/LLMs/
```

    [NbConvertApp] Converting notebook LLMs.ipynb to markdown
    [NbConvertApp] Writing 39952 bytes to Format/LLMs/LLMs.md


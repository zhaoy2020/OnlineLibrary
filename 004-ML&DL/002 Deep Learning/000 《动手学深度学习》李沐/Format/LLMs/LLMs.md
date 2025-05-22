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
- 10. [调用大语言模型](#toc10_)    
  - 10.1. [OpenAI Python SDK](#toc10_1_)    
    - 10.1.1. [文本生成](#toc10_1_1_)    
    - 10.1.2. [代码补全](#toc10_1_2_)    
    - 10.1.3. [图像生成（DALL-E）](#toc10_1_3_)    
    - 10.1.4. [图像识别](#toc10_1_4_)    
    - 10.1.5. [语音转文本（Whisper）](#toc10_1_5_)    
    - 10.1.6. [错误处理与最佳实践](#toc10_1_6_)    
    - 10.1.7. [openai库的高级用法](#toc10_1_7_)    
      - 10.1.7.1. [异步支持](#toc10_1_7_1_)    
      - 10.1.7.2. [微调（Fine-tuning）：](#toc10_1_7_2_)    
      - 10.1.7.3. [流式响应：](#toc10_1_7_3_)    
  - 10.2. [deepseek-ai的SDK](#toc10_2_)    
  - 10.3. [curl接口](#toc10_3_)    
- 11. [部署大模型](#toc11_)    
  - 11.1. [下载模型](#toc11_1_)    
  - 11.2. [ollama](#toc11_2_)    
    - 11.2.1. [Install and run model](#toc11_2_1_)    
    - 11.2.2. [API on web port](#toc11_2_2_)    
    - 11.2.3. [Python ollama module](#toc11_2_3_)    
      - 11.2.3.1. [demo：翻译中文为英文](#toc11_2_3_1_)    
  - 11.3. [ktransformers](#toc11_3_)    
    - 11.3.1. [Docker安装](#toc11_3_1_)    
    - 11.3.2. [编译安装](#toc11_3_2_)    
      - 11.3.2.1. [prepare](#toc11_3_2_1_)    
      - 11.3.2.2. [方式一：pip3 install whl](#toc11_3_2_2_)    
      - 11.3.2.3. [方式二：编译](#toc11_3_2_3_)    
    - 11.3.3. [虚拟机中编译安装](#toc11_3_3_)    
    - 11.3.4. [使用](#toc11_3_4_)    
- 12. [启动子预测](#toc12_)    
- 13. [转格式](#toc13_)    

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

# 10. <a id='toc10_'></a>[调用大语言模型](#toc0_)

## 10.1. <a id='toc10_1_'></a>[OpenAI Python SDK](#toc0_)

### 10.1.1. <a id='toc10_1_1_'></a>[文本生成](#toc0_)


```python
# Please install OpenAI SDK first: `pip3 install openai`

import os
import openai 


client = openai.OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"), 
    base_url = "https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False, 
    max_tokens=2048,
)

print(response.choices[0].message.content)

# # stream=True的时候，启用流示返回
# for chunk in response:
#     print(chunk.choices[0].delta.content, end="", flush=True)
```


    ---------------------------------------------------------------------------

    OpenAIError                               Traceback (most recent call last)

    Cell In[5], line 7
          3 import os
          4 import openai 
    ----> 7 client = openai.OpenAI(
          8     api_key = os.getenv("OPENAI_API_KEY"), 
          9     base_url = "https://api.deepseek.com/v1"
         10 )
         12 response = client.chat.completions.create(
         13     model="deepseek-chat",
         14     messages=[
       (...)
         18     stream=False
         19 )
         21 print(response.choices[0].message.content)


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_client.py:110, in OpenAI.__init__(self, api_key, organization, project, base_url, websocket_base_url, timeout, max_retries, default_headers, default_query, http_client, _strict_response_validation)
        108     api_key = os.environ.get("OPENAI_API_KEY")
        109 if api_key is None:
    --> 110     raise OpenAIError(
        111         "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
        112     )
        113 self.api_key = api_key
        115 if organization is None:


    OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable


### 10.1.2. <a id='toc10_1_2_'></a>[代码补全](#toc0_)


```python
response = client.chat.completions.create(
    messages = [
        {'role': 'user', 'content': "1 + 1"},
    ],
    model = "gpt-3.5-turbo", 
    stream = False, 
    max_tokens = 2048,
)

print(response.choices[0].message.content)
```

### 10.1.3. <a id='toc10_1_3_'></a>[图像生成（DALL-E）](#toc0_)

调用images.generate生成图像：


```python
response = client.images.generate(
    prompt="一只穿着宇航服的猫",
    n=1,
    size="1024x1024"
)

print(response.data.url)

```

### 10.1.4. <a id='toc10_1_4_'></a>[图像识别](#toc0_)


```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "这是什么？",
                    "type": "text"
                },
                {
                    "image_url": {
                        "url": "data:image/png;base64,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"
                    },
                    "type": "image_url"
                }
            ]
        }
    ],
    model='gpt-4o-2024-05-13',
    stream=False,
    max_tokens=200
)

print(response.choices[0].message.content)
```

### 10.1.5. <a id='toc10_1_5_'></a>[语音转文本（Whisper）](#toc0_)

使用audio.transcriptions处理音频文件：


```python
audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

print(transcript.text)

```

### 10.1.6. <a id='toc10_1_6_'></a>[错误处理与最佳实践](#toc0_)

异常捕获：处理API请求中的常见错误（如认证失败、超时）：


```python
from openai import APIError
try:
    response = client.chat.completions.create(...)
except APIError as e:
    print(f"API请求失败: {e}")

```

### 10.1.7. <a id='toc10_1_7_'></a>[openai库的高级用法](#toc0_)

#### 10.1.7.1. <a id='toc10_1_7_1_'></a>[异步支持](#toc0_)

SDK提供了异步客户端AsyncOpenAI,使用方法与同步客户端类似,只需在API调用前加上await:


```python

```

#### 10.1.7.2. <a id='toc10_1_7_2_'></a>[微调（Fine-tuning）：](#toc0_)

支持对基础模型进行定制化训练，需准备数据集并调用fine_tuning.jobs.create

#### 10.1.7.3. <a id='toc10_1_7_3_'></a>[流式响应：](#toc0_)

通过stream=True实现逐词实时输出，适用于交互式场景：


```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    stream=True
)

for chunk in stream:
    print(chunk.choices.delta.content or "", end="")

```

## 10.2. <a id='toc10_2_'></a>[deepseek-ai的SDK](#toc0_)

- Temperature 设置，参数默认为 1.0。

|场景	|温度|
|-|-|
|代码生成/数学解题   	|0.0|
|数据抽取/分析	|1.0|
|通用对话	|1.3|
|翻译	|1.3|
|创意类写作/诗歌创作	|1.5|


```python
# Please install OpenAI SDK first: `pip3 install openai`

import os 
from openai import OpenAI 


client = OpenAI(
    api_key = "sk-eb45f6f0191d432686061f66e3eb4f29", 
    base_url = "https://api.deepseek.com/v1",            # 出于与 OpenAI 兼容考虑，此处 v1 与模型版本无关。
)

```


```python
print(client.models.list())
```

    SyncPage[Model](data=[Model(id='deepseek-chat', created=None, object='model', owned_by='deepseek'), Model(id='deepseek-reasoner', created=None, object='model', owned_by='deepseek')], object='list')



```python


response = client.chat.completions.create(
    # model="deepseek-chat",        # DeepSeek-V3
    model = "deepseek-reasoner",    # DeepSeek 最新推出的推理模型 DeepSeek-R1
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```


    ---------------------------------------------------------------------------

    APIStatusError                            Traceback (most recent call last)

    Cell In[15], line 1
    ----> 1 response = client.chat.completions.create(
          2     # model="deepseek-chat",        # DeepSeek-V3
          3     model = "deepseek-reasoner",    # DeepSeek 最新推出的推理模型 DeepSeek-R1
          4     messages=[
          5         {"role": "system", "content": "You are a helpful assistant"},
          6         {"role": "user", "content": "Hello"},
          7     ],
          8     stream=False
          9 )
         11 print(response.choices[0].message.content)


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_utils/_utils.py:279, in required_args.<locals>.inner.<locals>.wrapper(*args, **kwargs)
        277             msg = f"Missing required argument: {quote(missing[0])}"
        278     raise TypeError(msg)
    --> 279 return func(*args, **kwargs)


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/resources/chat/completions.py:863, in Completions.create(self, messages, model, audio, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, modalities, n, parallel_tool_calls, prediction, presence_penalty, reasoning_effort, response_format, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, extra_headers, extra_query, extra_body, timeout)
        821 @required_args(["messages", "model"], ["messages", "model", "stream"])
        822 def create(
        823     self,
       (...)
        860     timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
        861 ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        862     validate_response_format(response_format)
    --> 863     return self._post(
        864         "/chat/completions",
        865         body=maybe_transform(
        866             {
        867                 "messages": messages,
        868                 "model": model,
        869                 "audio": audio,
        870                 "frequency_penalty": frequency_penalty,
        871                 "function_call": function_call,
        872                 "functions": functions,
        873                 "logit_bias": logit_bias,
        874                 "logprobs": logprobs,
        875                 "max_completion_tokens": max_completion_tokens,
        876                 "max_tokens": max_tokens,
        877                 "metadata": metadata,
        878                 "modalities": modalities,
        879                 "n": n,
        880                 "parallel_tool_calls": parallel_tool_calls,
        881                 "prediction": prediction,
        882                 "presence_penalty": presence_penalty,
        883                 "reasoning_effort": reasoning_effort,
        884                 "response_format": response_format,
        885                 "seed": seed,
        886                 "service_tier": service_tier,
        887                 "stop": stop,
        888                 "store": store,
        889                 "stream": stream,
        890                 "stream_options": stream_options,
        891                 "temperature": temperature,
        892                 "tool_choice": tool_choice,
        893                 "tools": tools,
        894                 "top_logprobs": top_logprobs,
        895                 "top_p": top_p,
        896                 "user": user,
        897             },
        898             completion_create_params.CompletionCreateParams,
        899         ),
        900         options=make_request_options(
        901             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        902         ),
        903         cast_to=ChatCompletion,
        904         stream=stream or False,
        905         stream_cls=Stream[ChatCompletionChunk],
        906     )


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_base_client.py:1283, in SyncAPIClient.post(self, path, cast_to, body, options, files, stream, stream_cls)
       1269 def post(
       1270     self,
       1271     path: str,
       (...)
       1278     stream_cls: type[_StreamT] | None = None,
       1279 ) -> ResponseT | _StreamT:
       1280     opts = FinalRequestOptions.construct(
       1281         method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1282     )
    -> 1283     return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_base_client.py:960, in SyncAPIClient.request(self, cast_to, options, remaining_retries, stream, stream_cls)
        957 else:
        958     retries_taken = 0
    --> 960 return self._request(
        961     cast_to=cast_to,
        962     options=options,
        963     stream=stream,
        964     stream_cls=stream_cls,
        965     retries_taken=retries_taken,
        966 )


    File ~/miniconda3/envs/pytorch/lib/python3.12/site-packages/openai/_base_client.py:1064, in SyncAPIClient._request(self, cast_to, options, retries_taken, stream, stream_cls)
       1061         err.response.read()
       1063     log.debug("Re-raising status error")
    -> 1064     raise self._make_status_error_from_response(err.response) from None
       1066 return self._process_response(
       1067     cast_to=cast_to,
       1068     options=options,
       (...)
       1072     retries_taken=retries_taken,
       1073 )


    APIStatusError: Error code: 402 - {'error': {'message': 'Insufficient Balance', 'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}


## 10.3. <a id='toc10_3_'></a>[curl接口](#toc0_)


```bash
%%bash 
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <DeepSeek API Key>" \
  -d '{
        "model": "deepseek-chat",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "stream": false
      }'
```

# 11. <a id='toc11_'></a>[部署大模型](#toc0_)


## 11.1. <a id='toc11_1_'></a>[下载模型](#toc0_)


```python
# https://huggingface.co/deepseek-ai/DeepSeek-R1
hfd.sh deepseek-ai/DeepSeek-R1 -x 10 -j 10 

# https://huggingface.co/unsloth/DeepSeek-R1-GGUF
hfd.sh unsloth/DeepSeek-R1-GGUF -x 10 -j 10 --include DeepSeek-R1-Q8_0

```

## 11.2. <a id='toc11_2_'></a>[ollama](#toc0_)
### 11.2.1. <a id='toc11_2_1_'></a>[Install and run model](#toc0_)


```python
# start the serve
ollama serve

# list all model images
ollama list 

# run model from image
ollama run model_card
```

### 11.2.2. <a id='toc11_2_2_'></a>[API on web port](#toc0_)
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

### 11.2.3. <a id='toc11_2_3_'></a>[Python ollama module](#toc0_)


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

#### 11.2.3.1. <a id='toc11_2_3_1_'></a>[demo：翻译中文为英文](#toc0_)


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


## 11.3. <a id='toc11_3_'></a>[ktransformers](#toc0_)


### 11.3.1. <a id='toc11_3_1_'></a>[Docker安装](#toc0_)

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

### 11.3.2. <a id='toc11_3_2_'></a>[编译安装](#toc0_)

[https://kvcache-ai.github.io/ktransformers/en/install.html](https://kvcache-ai.github.io/ktransformers/en/install.html)

#### 11.3.2.1. <a id='toc11_3_2_1_'></a>[prepare](#toc0_)


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

#### 11.3.2.2. <a id='toc11_3_2_2_'></a>[方式一：pip3 install whl](#toc0_)


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

#### 11.3.2.3. <a id='toc11_3_2_3_'></a>[方式二：编译](#toc0_)


```python
# Make sure your system has dual sockets and double size RAM than the model's size (e.g. 1T RAM for 512G model)
export USE_NUMA=1
bash install.sh # or `make dev_install`
```

### 11.3.3. <a id='toc11_3_3_'></a>[虚拟机中编译安装](#toc0_)

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

### 11.3.4. <a id='toc11_3_4_'></a>[使用](#toc0_)


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

# 12. <a id='toc12_'></a>[启动子预测](#toc0_)


```python

```

# 13. <a id='toc13_'></a>[转格式](#toc0_)


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

    [NbConvertApp] Making directory ./Format/LLMs
    [NbConvertApp] Converting notebook LLMs.ipynb to html
    [NbConvertApp] Writing 405847 bytes to Format/LLMs/LLMs.html



```python
# ipynb to markdown
!jupyter nbconvert --to markdown LLMs.ipynb --output-dir=./Format/LLMs/
```

    [NbConvertApp] Converting notebook LLMs.ipynb to markdown
    [NbConvertApp] Writing 40263 bytes to Format/LLMs/LLMs.md


# 一、基于PaddleNLP的美国大选的新闻真假分类（二）基于SKEP模型的分类任务

## 0.解释
**本来这个都烂尾了，看到有人问二在哪儿？只好说还没公开，自己挖的坑，含泪也要填。下次标题再也不屑一、二了，真的很容易烂尾。。。。。。**

## 1.简介
新闻媒体已成为向世界人民传递世界上正在发生的事情的信息的渠道。 人们通常认为新闻中传达的一切都是真实的。 在某些情况下，甚至新闻频道也承认他们的新闻不如他们写的那样真实。 但是，一些新闻不仅对人民或政府产生重大影响，而且对经济也产生重大影响。 一则新闻可以根据人们的情绪和政治局势上下移动曲线。

从真实的真实新闻中识别虚假新闻非常重要。 该问题已通过自然语言处理工具解决并得到了解决，本篇文章可帮助我们根据历史数据识别假新闻或真实新闻。

## 2.问题描述
对于印刷媒体和数字媒体，信息的真实性已成为影响企业和社会的长期问题。在社交网络上，信息传播的范围和影响以如此快的速度发生，并且如此迅速地放大，以至于失真，不准确或虚假的信息具有巨大的潜力，可在数分钟内对数百万用户造成现实世界的影响。最近，人们表达了对该问题的一些担忧，并提出了一些缓解该问题的方法。

在各种信息广播的整个历史中，一直存在着不那么精确的引人注目和引人入胜的新闻标题，这些新闻标题旨在吸引观众的注意力来出售信息。但是，在社交网站上，信息传播的范围和影响得到了显着放大，并且发展速度如此之快，以至于失真，不准确或虚假的信息具有巨大的潜力，可在数分钟内为数百万的用户带来真正的影响。

## 3.目标
* 我们唯一的目标是将数据集中的新闻分类为假新闻或真实新闻。
* 新闻的细致EDA
* 选择并建立强大的分类模型
## 数据
数据地址：[https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

# 二、数据处理

## 1.PaddleNLP环境


```python
!pip install -U paddlenlp
```

## 2.解压缩数据


```python
# 运行一次解压缩，后续注释掉
# !unzip data/data27271/真假新闻数据集.zip
```

## 3.导出基础库


```python
# 基本数据包：pandas和numpy
import pandas as pd 
import numpy as np 
import os
import paddle
import paddle.nn.functional as F
```

## 4.加载数据


```python
import pandas as pd
# 读取数据集
fake_news = pd.read_csv('Fake.csv')
true_news = pd.read_csv('True.csv')
# 虚假新闻数据集的大小以及字段
print ("虚假新闻数据集的大小以及字段 (row, column):"+ str(fake_news.shape))
print (fake_news.info())
print("\n --------------------------------------- \n")
# 真实新闻数据集的大小以及字段
print ("真实新闻数据集的大小以及字段 (row, column):"+ str(true_news.shape))
print (true_news.info())
```

    虚假新闻数据集的大小以及字段 (row, column):(23481, 4)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23481 entries, 0 to 23480
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   title    23481 non-null  object
     1   text     23481 non-null  object
     2   subject  23481 non-null  object
     3   date     23481 non-null  object
    dtypes: object(4)
    memory usage: 733.9+ KB
    None
    
     --------------------------------------- 
    
    真实新闻数据集的大小以及字段 (row, column):(21417, 4)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21417 entries, 0 to 21416
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   title    21417 non-null  object
     1   text     21417 non-null  object
     2   subject  21417 non-null  object
     3   date     21417 non-null  object
    dtypes: object(4)
    memory usage: 669.4+ KB
    None


## 5.数据合并



```python
# 标签转换
true_news['label'] = 0
fake_news['label'] = 1

# 数据合并
news_all = pd.concat([true_news, fake_news], ignore_index=True)
```


```python
news_all.info
```




    <bound method DataFrame.info of                                                    title  \
    0      As U.S. budget fight looms, Republicans flip t...   
    1      U.S. military to accept transgender recruits o...   
    2      Senior U.S. Republican senator: 'Let Mr. Muell...   
    3      FBI Russia probe helped by Australian diplomat...   
    4      Trump wants Postal Service to charge 'much mor...   
    ...                                                  ...   
    44893  McPain: John McCain Furious That Iran Treated ...   
    44894  JUSTICE? Yahoo Settles E-mail Privacy Class-ac...   
    44895  Sunnistan: US and Allied ‘Safe Zone’ Plan to T...   
    44896  How to Blow $700 Million: Al Jazeera America F...   
    44897  10 U.S. Navy Sailors Held by Iranian Military ...   
    
                                                        text       subject  \
    0      WASHINGTON (Reuters) - The head of a conservat...  politicsNews   
    1      WASHINGTON (Reuters) - Transgender people will...  politicsNews   
    2      WASHINGTON (Reuters) - The special counsel inv...  politicsNews   
    3      WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   
    4      SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   
    ...                                                  ...           ...   
    44893  21st Century Wire says As 21WIRE reported earl...   Middle-east   
    44894  21st Century Wire says It s a familiar theme. ...   Middle-east   
    44895  Patrick Henningsen  21st Century WireRemember ...   Middle-east   
    44896  21st Century Wire says Al Jazeera America will...   Middle-east   
    44897  21st Century Wire says As 21WIRE predicted in ...   Middle-east   
    
                         date  label  
    0      December 31, 2017       0  
    1      December 29, 2017       0  
    2      December 31, 2017       0  
    3      December 30, 2017       0  
    4      December 29, 2017       0  
    ...                   ...    ...  
    44893    January 16, 2016      1  
    44894    January 16, 2016      1  
    44895    January 15, 2016      1  
    44896    January 14, 2016      1  
    44897    January 12, 2016      1  
    
    [44898 rows x 5 columns]>



## 6.数据集划分


```python
# 自定义reader方法
from paddlenlp.datasets import load_dataset
from paddle.io import Dataset, Subset
from paddlenlp.datasets import MapDataset

def read(pd_data):
    for index, item in pd_data.iterrows():       
        yield {'text': item['title']+'. '+item['text'], 'label': item['label'], 'qid': index}
```


```python
# 划分数据集
all_ds = load_dataset(read, pd_data=news_all,lazy=False)
train_ds = Subset(dataset=all_ds, indices=[i for i in range(len(all_ds)) if i % 10 != 1])
dev_ds = Subset(dataset=all_ds, indices=[i for i in range(len(all_ds)) if i % 10 == 1])

# 在转换为MapDataset类型
train_ds = MapDataset(train_ds)
dev_ds = MapDataset(dev_ds)
print(len(train_ds))
print(len(dev_ds))
```

    40408
    4490


# 三、SKEP模型加载


```python
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

# 指定模型名称，一键加载模型
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en", num_classes=2)
# 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等。
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")
```

    [2021-07-26 00:43:18,015] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_2.0_large_en/skep_ernie_2.0_large_en.pdparams
    [2021-07-26 00:43:28,446] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_2.0_large_en/skep_ernie_2.0_large_en.vocab.txt


# 四、NLP数据处理

## 1.加入日志


```python
# visualdl引入
from visualdl import LogWriter

writer = LogWriter("./log")
```

## 2.SkepTokenizer数据处理
SKEP模型对文本处理按照字粒度进行处理，我们可以使用PaddleNLP内置的SkepTokenizer完成一键式处理。


```python
import os
from functools import partial


import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

from utils import create_dataloader

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
   
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid
```


```python
# 批量数据大小
batch_size = 10
# 文本序列最大长度
max_seq_length = 512

# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

# 五、模型训练和评估

## 1.开始训练


```python
import time

from utils import evaluate

# 训练轮次
epochs = 3
# 训练过程中保存模型参数的文件夹
ckpt_dir = "skep_ckpt"
# len(train_data_loader)一轮训练所需要的step数
num_training_steps = len(train_data_loader) * epochs

# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=2e-5,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy评价指标
metric = paddle.metric.Accuracy()
```


```python
# 开启训练
global_step = 0
tic_train = time.time()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, labels = batch
        # 喂数据给model
        logits = model(input_ids, token_type_ids)
        # 计算损失函数值
        loss = criterion(logits, labels)
        # 预测分类概率值
        probs = F.softmax(logits, axis=1)
        # 计算acc
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        
        # 反向梯度回传，更新参数
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            save_dir = os.path.join(ckpt_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 评估当前训练的模型
            evaluate(model, criterion, metric, dev_data_loader)
            # 保存当前模型参数等
            model.save_pretrained(save_dir)
            # 保存tokenizer的词表等
            tokenizer.save_pretrained(save_dir)
```

## 2.训练日志
```
global step 110, epoch: 1, batch: 110, loss: 0.00653, accu: 0.98000, speed: 0.05 step/s
global step 120, epoch: 1, batch: 120, loss: 0.00180, accu: 0.99000, speed: 0.95 step/s
global step 130, epoch: 1, batch: 130, loss: 0.00236, accu: 0.99000, speed: 0.94 step/s
global step 140, epoch: 1, batch: 140, loss: 0.00210, accu: 0.99250, speed: 0.94 step/s
global step 150, epoch: 1, batch: 150, loss: 0.00216, accu: 0.99400, speed: 0.95 step/s
global step 160, epoch: 1, batch: 160, loss: 0.00651, accu: 0.99500, speed: 0.95 step/s
global step 170, epoch: 1, batch: 170, loss: 0.00105, accu: 0.99571, speed: 0.95 step/s
global step 180, epoch: 1, batch: 180, loss: 0.00092, accu: 0.99625, speed: 0.94 step/s
global step 190, epoch: 1, batch: 190, loss: 0.00065, accu: 0.99667, speed: 0.94 step/s
global step 200, epoch: 1, batch: 200, loss: 0.00058, accu: 0.99700, speed: 0.95 step/s
eval loss: 0.00571, accu: 0.99866
```

# 六、总结
## 1.数据处理
* 合并数据集
* 生成新标签
* 数据集划分
## 2.PaddleNLP自定义reader方法
以前多是直接读文件再返回，最近一直用pandas读取返回，更方便快捷
## 3.SKEP模型应用
分类好像就这一个模型，max_seq_length指的是单词最大数量，不超过512，如果超过了要用各种trick，比如前面截取、后面截取，诸如此类，总之会丢掉一部分。

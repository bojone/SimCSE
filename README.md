# SimCSE 中文测试

SimCSE在常见中文数据集上的测试，包含[ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC)、[BQ](http://icrc.hitsz.edu.cn/info/1037/1162.htm)、[LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html)、[PAWSX](https://arxiv.org/abs/1908.11828)、[STS-B](https://github.com/pluto-junzeng/CNSD)共5个任务。

## 介绍

- 博客：https://kexue.fm/archives/8348
- 论文：[《SimCSE: Simple Contrastive Learning of Sentence Embeddings》](https://arxiv.org/abs/2104.08821)
- 官方：https://github.com/princeton-nlp/SimCSE

## 文件

```
- utils.py  工具函数
- eval.py  评测主文件
```

## 评测

命令格式：
```
python eval.py [model_type] [pooling] [task_name] [dropout_rate]
```

使用例子：
```
python eval.py BERT cls ATEC 0.3
```

其中四个参数必须传入，含义分别如下：
```
- model_type: 模型，必须是['BERT', 'RoBERTa', 'WoBERT', 'RoFormer', 'BERT-large', 'RoBERTa-large', 'SimBERT', 'SimBERT-tiny', 'SimBERT-small']之一；
- pooling: 池化方式，必须是['first-last-avg', 'last-avg', 'cls', 'pooler']之一；
- task_name: 评测数据集，必须是['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']之一；
- dropout_rate: 浮点数，dropout的比例，如果为0则不dropout；
```

## 环境
测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.5，如果在其他环境组合下报错，请根据错误信息自行调整代码。

## 下载

Google官方的两个BERT模型：
- BERT：[chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)
- RoBERTa：[chinese_roberta_wwm_ext_L-12_H-768_A-12.zip](https://github.com/ymcui/Chinese-BERT-wwm)
- NEZHA：[NEZHA-base-WWM](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)
- WoBERT：[chinese_wobert_plus_L-12_H-768_A-12.zip](https://github.com/ZhuiyiTechnology/WoBERT)
- RoFormer：[chinese_roformer_L-12_H-768_A-12.zip](https://github.com/ZhuiyiTechnology/roformer)
- SimBERT: [chinese_simbert_L-12_H-768_A-12.zip](https://github.com/ZhuiyiTechnology/simbert)
- SimBERT-small: [chinese_simbert_L-6_H-384_A-12.zip](https://github.com/ZhuiyiTechnology/simbert)
- SimBERT-tiny: [chinese_simbert_L-4_H-312_A-12.zip](https://github.com/ZhuiyiTechnology/simbert)

关于语义相似度数据集，可以从数据集对应的链接自行下载，也可以从作者提供的百度云链接下载。
- 链接: https://pan.baidu.com/s/1d6jSiU1wHQAEMWJi7JJWCQ 提取码: qkt6

其中senteval_cn目录是评测数据集汇总，senteval_cn.zip是senteval目录的打包，两者下其一就好。

## 相关
- BERT-whitening：https://github.com/bojone/BERT-whitening

## 交流

QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn

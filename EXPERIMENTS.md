# EE6483 补充实验与交稿指南

## CIFAR-10（g、h）

[查看四组正式结果](CIFAR10_RESULTS.md)。

CIFAR-10 的输入是 32×32 RGB 图，任务从猫狗二分类扩展为十分类。保留 ImageNet 预训练 ResNet-18 和先冻结再微调的训练流程，将分类头改为 512→10。训练图可随机裁剪和水平翻转，随后缩放到 96×96；验证与测试只使用固定缩放和归一化。

从官方 50,000 张训练图中按类固定留出 500 张作验证。均衡实验剩余 45,000 张训练图；不平衡实验将 cat、deer、dog 各缩减到 450 张，其余类仍各 4,500 张。三种不平衡设置共享同一训练子集、验证集和训练预算。先按验证准确率选择检查点，再评估官方 10,000 张测试图。

在仓库根目录运行：

```powershell
python src/train.py --config configs/cifar10.yaml --eval-test
python src/train.py --config configs/cifar10.yaml --eval-test --set 'minority_classes=[3,4,5]' exp_name=cifar10_imbalanced
python src/train.py --config configs/cifar10.yaml --eval-test --set 'minority_classes=[3,4,5]' imbalance_method=weighted_loss exp_name=cifar10_weighted
python src/train.py --config configs/cifar10.yaml --eval-test --set 'minority_classes=[3,4,5]' imbalance_method=oversample exp_name=cifar10_oversample
```

- **加权交叉熵**：训练时按类别频数的倒数提高少数类错误的代价，不改变抽到哪些样本。
- **过采样**：按类别频数的倒数设定每张图的抽样权重，有放回地抽取，保持每轮抽样总次数不变。少数类重复出现时会应用随机增强。
- 验证和测试均保持原分布；报告中从各组 `metrics.json` 取测试准确率、宏平均 F1、各类召回率和混淆矩阵。用三个少数类的召回率说明两种策略分别改善了哪些类，以及多数类是否有损失。不要用测试集调参。

## 猫狗分类（b、d、e、f）

ResNet-18 的结构图见 [架构图](docs/resnet18.svg)。提交报告时同时提供现有训练/推理代码和 README 中的安装步骤。

当前提交来源于原有 `resnet18_ft_s42`。如果最终选用 99.02% 的模型，并且手头有该次实验权重，可重新生成。建议先另存以便对照：

```powershell
python src/predict.py --ckpt outputs/r18_lr5e-4_s42/best.pt --data-root datasets --out submission_final.csv
python src/analyze.py --run outputs/r18_lr5e-4_s42 --out outputs/report_final --test-dir datasets/test
```

没有该次权重时可用实验目录的配置重训，使用不同实验名保护已有结果。重训结果应引用新跑出的指标，避免假定能得到完全相同的数值。

也可以直接分析仓库里已有的结果：

```powershell
python src/analyze.py --run outputs/resnet18_ft_s42 --out outputs/report_existing
```

如已准备课程测试图片，在上面的命令末尾加 `--test-dir datasets/test` 生成候选图。人工检查候选，选出 1–2 张有代表性的样例。记录图片 ID、模型预测、概率、人工判断和分析。低置信度候选不等同于错例；验证集的标签也不能用来替代测试图的人工作答。

数据处理对照：固定验证集、种子和其余训练设置，先完成全量有/无增强对照。再用同一组分层抽样补两个 2k 实验：

```powershell
python src/train.py --config configs/resnet18_ft.yaml --set lr=5e-4 augment=false exp_name=r18_20k_noaug
python src/train.py --config configs/resnet18_ft.yaml --set lr=5e-4 train_per_class=1000 exp_name=r18_2k_aug
python src/train.py --config configs/resnet18_ft.yaml --set lr=5e-4 train_per_class=1000 augment=false exp_name=r18_2k_noaug
```



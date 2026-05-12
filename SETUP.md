# AutoDL 实验部署说明

## 目录结构

克隆仓库并解压数据集后，目录结构如下：

```
/root/autodl-tmp/AML_Assignment/
├── src/
│   ├── model.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tent.py
│   ├── eata.py
│   ├── cr_tta.py
│   └── run_all.py
├── TableShift_Dataset/          ← 解压后生成
│   ├── assistments/
│   ├── nhanes_lead/
│   ├── brfss_diabetes/
│   ├── acsfoodstamps/
│   ├── physionet/
│   └── acsunemployment/
├── results/                     ← 运行后自动生成
├── TableShift_Dataset.zip
├── requirements.txt
└── SETUP.md
```

---

## 第一步：克隆仓库

```bash
cd /root/autodl-tmp
git clone https://github.com/<你的用户名>/AML_Assignment.git
cd AML_Assignment
```

> 仓库使用 Git LFS 存储数据集压缩包，clone 时会自动下载，无需额外操作。

---

## 第二步：解压数据集

```bash
cd /root/autodl-tmp/AML_Assignment
unzip TableShift_Dataset.zip
```

验证解压是否正确：

```bash
ls TableShift_Dataset/
# 应看到：assistments  nhanes_lead  brfss_diabetes  acsfoodstamps  physionet  acsunemployment
```

---

## 第三步：安装依赖

```bash
pip install -r requirements.txt
```

> AutoDL 镜像已预装 CUDA 版 PyTorch，pip install 会复用已有版本，通常只需安装 pandas / scikit-learn。

---

## 第四步：运行实验

建议在 tmux 中运行，防止断连中断：

```bash
tmux new -s exp
```

### 服务器 1（assistments / brfss_diabetes / nhanes_lead）

```bash
cd /root/autodl-tmp/AML_Assignment
python src/run_all.py \
    --datasets assistments brfss_diabetes nhanes_lead \
    --seeds 0 1 2
```

### 服务器 2（acsunemployment / physionet / acsfoodstamps）

```bash
cd /root/autodl-tmp/AML_Assignment
python src/run_all.py \
    --datasets acsunemployment physionet acsfoodstamps \
    --seeds 0 1 2
```

> tmux 挂后台：`Ctrl+B` 然后按 `D`；重新连接：`tmux attach -t exp`

---

## 第五步：合并两台服务器的结果

两台服务器分别下载各自的 `results/results.csv`，在本地合并：

```python
import pandas as pd

df1 = pd.read_csv("results_server1.csv")
df2 = pd.read_csv("results_server2.csv")
df  = pd.concat([df1, df2], ignore_index=True)
df  = df.drop_duplicates(subset=["dataset", "seed", "method"], keep="last")
df  = df.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
df.to_csv("results_all.csv", index=False)
print(df)
```

---

## 预估运行时间（AutoDL A5000/3090，GPU）

| 数据集 | 训练行数 | 预估单 seed 时间 |
|---|---|---|
| nhanes_lead | 11,807 | ~2 分钟 |
| brfss_diabetes | 969,229 | ~10 分钟 |
| assistments | 2,132,526 | ~20 分钟 |
| acsfoodstamps | 629,018 | ~8 分钟 |
| physionet | 1,122,299 | ~12 分钟 |
| acsunemployment | 1,290,914 | ~15 分钟 |

每台服务器跑 3 个数据集 × 3 seed，总预估约 **2~3 小时**。

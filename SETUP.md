# AutoDL 瀹為獙閮ㄧ讲璇存槑

## 鐩綍缁撴瀯

鍏嬮殕浠撳簱骞惰В鍘嬫暟鎹泦鍚庯紝鐩綍缁撴瀯濡備笅锛?

```
/root/autodl-tmp/AML_NJU/
鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ model.py
鈹?  鈹溾攢鈹€ data_loader.py
鈹?  鈹溾攢鈹€ train.py
鈹?  鈹溾攢鈹€ evaluate.py
鈹?  鈹溾攢鈹€ tent.py
鈹?  鈹溾攢鈹€ eata.py
鈹?  鈹溾攢鈹€ cr_tta.py
鈹?  鈹斺攢鈹€ run_all.py
鈹溾攢鈹€ TableShift_Dataset/          鈫?瑙ｅ帇鍚庣敓鎴?
鈹?  鈹溾攢鈹€ assistments/
鈹?  鈹溾攢鈹€ nhanes_lead/
鈹?  鈹溾攢鈹€ brfss_diabetes/
鈹?  鈹溾攢鈹€ acsfoodstamps/
鈹?  鈹溾攢鈹€ physionet/
鈹?  鈹斺攢鈹€ acsunemployment/
鈹溾攢鈹€ results/                     鈫?杩愯鍚庤嚜鍔ㄧ敓鎴?
鈹溾攢鈹€ TableShift_Dataset.zip
鈹溾攢鈹€ requirements.txt
鈹斺攢鈹€ SETUP.md
```

---

## 绗竴姝ワ細鍏嬮殕浠撳簱

```bash
cd /root/autodl-tmp
git clone https://github.com/SleetZhang/AML_NJU.git
cd AML_NJU
```

> 浠撳簱浣跨敤 Git LFS 瀛樺偍鏁版嵁闆嗗帇缂╁寘锛宑lone 鏃朵細鑷姩涓嬭浇锛屾棤闇€棰濆鎿嶄綔銆?

---

## 绗簩姝ワ細瑙ｅ帇鏁版嵁闆?

```bash
cd /root/autodl-tmp/AML_NJU
unzip TableShift_Dataset.zip
```

楠岃瘉瑙ｅ帇鏄惁姝ｇ‘锛?

```bash
ls TableShift_Dataset/
# 搴旂湅鍒帮細assistments  nhanes_lead  brfss_diabetes  acsfoodstamps  physionet  acsunemployment
```

---

## 绗笁姝ワ細瀹夎渚濊禆

```bash
pip install -r requirements.txt
```

> AutoDL 闀滃儚宸查瑁?CUDA 鐗?PyTorch锛宲ip install 浼氬鐢ㄥ凡鏈夌増鏈紝閫氬父鍙渶瀹夎 pandas / scikit-learn銆?

---

## 绗洓姝ワ細杩愯瀹為獙

寤鸿鍦?tmux 涓繍琛岋紝闃叉鏂繛涓柇锛?

```bash
tmux new -s exp
```

### 鏈嶅姟鍣?1锛坅ssistments / brfss_diabetes / nhanes_lead锛?

```bash
cd /root/autodl-tmp/AML_NJU
python src/run_all.py \
    --datasets assistments brfss_diabetes nhanes_lead \
    --seeds 0 1 2
```

### 鏈嶅姟鍣?2锛坅csunemployment / physionet / acsfoodstamps锛?

```bash
cd /root/autodl-tmp/AML_NJU
python src/run_all.py \
    --datasets acsunemployment physionet acsfoodstamps \
    --seeds 0 1 2
```

> tmux 鎸傚悗鍙帮細`Ctrl+B` 鐒跺悗鎸?`D`锛涢噸鏂拌繛鎺ワ細`tmux attach -t exp`

---

## 绗簲姝ワ細鍚堝苟涓ゅ彴鏈嶅姟鍣ㄧ殑缁撴灉

涓ゅ彴鏈嶅姟鍣ㄥ垎鍒笅杞藉悇鑷殑 `results/results.csv`锛屽湪鏈湴鍚堝苟锛?

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

## 棰勪及杩愯鏃堕棿锛圓utoDL A5000/3090锛孏PU锛?

| 鏁版嵁闆?| 璁粌琛屾暟 | 棰勪及鍗?seed 鏃堕棿 |
|---|---|---|
| nhanes_lead | 11,807 | ~2 鍒嗛挓 |
| brfss_diabetes | 969,229 | ~10 鍒嗛挓 |
| assistments | 2,132,526 | ~20 鍒嗛挓 |
| acsfoodstamps | 629,018 | ~8 鍒嗛挓 |
| physionet | 1,122,299 | ~12 鍒嗛挓 |
| acsunemployment | 1,290,914 | ~15 鍒嗛挓 |

姣忓彴鏈嶅姟鍣ㄨ窇 3 涓暟鎹泦 脳 3 seed锛屾€婚浼扮害 **2~3 灏忔椂**銆?


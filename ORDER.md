# 学习顺序与练习清单 (Learning Order & Checklist)

按本清单从上到下学习：**基础 → 通用技能 → 通用知识**。`assessments/` 是按能力扁平存放的题库，顺序以本清单为准。图例：📖 读课程　✍️ 编程练习　🎯 项目　📝 小测。


## 基础 Foundations


### Python 工程  →  `curriculum/foundations/01-python/`
- 📖 课程：`curriculum/foundations/01-python/lessons/`（按顺序）
- ✍️ 编程练习：`w1-oop` · `w1-dataclass` · `w1-cli` · `w1-decorators` · `w1-generators`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w1-python.yaml`


### NumPy 与向量化  →  `curriculum/foundations/02-numpy/`
- 📖 课程：`curriculum/foundations/02-numpy/lessons/`（按顺序）
- ✍️ 编程练习：`w2-numpy` · `w2-broadcasting` · `w2-einsum` · `w2-ols` · `w2-softmax`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w2-numpy.yaml`


### pandas 与数据工程  →  `curriculum/foundations/03-pandas/`
- 📖 课程：`curriculum/foundations/03-pandas/lessons/`（按顺序）
- ✍️ 编程练习：`w3-groupby` · `w3-merge` · `w3-io` · `w3-clean`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 🎯 项目：`p-w3-data-pipeline` — 见 `projects/p-w3-data-pipeline/`；实现 `assessments/autograder/p-w3-data-pipeline/starter.py`
- 📝 小测：`assessments/quizzes/w3-pandas.yaml`


### 统计推断  →  `curriculum/foundations/04-statistics/`
- 📖 课程：`curriculum/foundations/04-statistics/lessons/`（按顺序）
- ✍️ 编程练习：`w4-descriptive` · `w4-clt` · `w4-ttest` · `w4-ols-inference` · `w4-multiple-testing`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w4-stats.yaml`


### 经典机器学习  →  `curriculum/foundations/05-classical-ml/`
- 📖 课程：`curriculum/foundations/05-classical-ml/lessons/`（按顺序）
- ✍️ 编程练习：`w5-cv` · `w5-logreg` · `w5-pca` · `w5-metrics` · `w5-xgb`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w5-ml.yaml`


### 深度学习入门 + nanograd L1–L3  →  `curriculum/foundations/06-deep-learning/`
- 📖 课程：`curriculum/foundations/06-deep-learning/lessons/`（按顺序）
- ✍️ 编程练习：`w6-torch-basics` · `w6-nn-module` · `w6-training-loop` · `w6-vae` · `nanograd-l1` · `nanograd-l2` · `nanograd-l3`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 🎯 项目：`p-w6-vae` — 见 `projects/p-w6-vae/`；实现 `assessments/autograder/p-w6-vae/starter.py`
- 📝 小测：`assessments/quizzes/w6-dl.yaml`


## 通用技能 · 深度学习 General Skills


### 张量自动微分 nanograd L4–L6  →  `curriculum/general-skills/01-tensor-autograd/`
- 📖 课程：`curriculum/general-skills/01-tensor-autograd/lessons/`（按顺序）
- ✍️ 编程练习：`nanograd-l4` · `nanograd-l5` · `nanograd-l6`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）


### Transformer  →  `curriculum/general-skills/02-transformers/`
- 📖 课程：`curriculum/general-skills/02-transformers/lessons/`（按顺序）
- ✍️ 编程练习：`w7-attention` · `w7-mha` · `w7-ft-transformer`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w7-transformers.yaml`


### 扩散模型与流匹配  →  `curriculum/general-skills/03-diffusion-flow/`
- 📖 课程：`curriculum/general-skills/03-diffusion-flow/lessons/`（按顺序）
- ✍️ 编程练习：`w9-ddpm` · `w9-flow-matching` · `w9-sampler`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w9-diffusion.yaml`


### 规模化训练  →  `curriculum/general-skills/04-training-at-scale/`
- 📖 课程：`curriculum/general-skills/04-training-at-scale/lessons/`（按顺序）
- ✍️ 编程练习：`w10-ema` · `w10-checkpoint` · `w10-grad-accum`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w10-scaling.yaml`


### 研究工程（可复现性与指标）  →  `curriculum/general-skills/05-research-engineering/`
- ✍️ 编程练习：`cap-repro` · `cap-metrics`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/cap-research-eng.yaml`


## 通用知识 · 统计遗传方法 General Knowledge


### 关联检验  →  `curriculum/general-knowledge/01-association-testing/`
- 📖 课程：`curriculum/general-knowledge/01-association-testing/lessons/`（按顺序）
- ✍️ 编程练习：`b7-maf` · `b7-assoc` · `b7-int` · `b7-gc`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w7b-gwas.yaml`


### 群体结构与混合模型  →  `curriculum/general-knowledge/02-population-structure-lmm/`
- 📖 课程：`curriculum/general-knowledge/02-population-structure-lmm/lessons/`（按顺序）
- ✍️ 编程练习：`b8-genopca` · `b8-ridge-lmm` · `b8-loco`（在 `assessments/autograder/<名>/starter.py` 里实现，`python check.py <名>` 自测）
- 📝 小测：`assessments/quizzes/w8b-lmm.yaml`
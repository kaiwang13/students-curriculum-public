# W3 数据管道小项目 `p-w3-data-pipeline`

对应能力 Competencies: **SH4**（数据清洗与规范化）、**SH5**（数据合并与输出）
通过线 Pass threshold: 自动评分 ≥ 0.7，量表每维度 ≥ 3/4

---

## 任务说明

你将使用合成表格数据，完成一条完整的数据管道：

| 步骤 | 内容 |
|------|------|
| 1. 生成脏数据 | 调用 `make_messy_table` 生成含重复行、缺失值、大小写混合 group 的原始数据 |
| 2. 探索性分析 | 用图表展示数据分布与脏数据特征（≥ 2 张带标注的 EDA 图） |
| 3. 实现管道 | 在 `solution.py` 中实现 `build_pipeline(df)` |
| 4. 输出存储 | 将清洗结果存为 `.parquet` 文件并验证读写一致 |

### `build_pipeline(df)` 接口规范

```python
def build_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """清洗合成表格数据：去重 → 数值列中位数填补缺失 → group 统一大写 → 重置索引。"""
```

具体要求：
- **去重**：删除完全重复的行
- **数值列 NaN 填补**：对所有 `dtype` 为数值型的列，用该列中位数填补缺失值
- **group 大写规范化**：`group` 列统一转换为大写（`"A"` / `"B"`）
- **重置索引**：返回的 DataFrame 索引从 0 开始连续编号

---

## 项目结构

```
progress/<你的名字>/
└── work/
    └── p-w3-data-pipeline/
        ├── solution.py        ← 实现 build_pipeline（必须）
        ├── analysis.ipynb     ← 探索与分析（EDA + 图表）
        └── clean.parquet      ← 清洗后数据（可选）
```

---

## 自动评分

```bash
# 从仓库根目录运行
python check.py p-w3-data-pipeline
```

通过条件：`score >= 0.7`（即 4 项测试中至少通过 2～3 项）。

评分项目：
1. `test_no_dupes` — 清洗后无重复行
2. `test_no_nans_in_numeric` — 数值列无 NaN
3. `test_group_normalized` — group 仅包含 `"A"` / `"B"`
4. `test_parquet_roundtrip` — parquet 存取结果一致

---

## 评分量表（导师评分）

详见 `assessments/rubrics/p-w3-data-pipeline.md`

维度：正确性 / 代码质量 / 评估严谨性 / 表达

---

## 参考资料

- W3 Lesson 04：数据清洗与规范化
- W3 Lesson 05：数据合并与输出
- W3 练习：`w3-clean`、`w3-io`
- pandas 文档：`drop_duplicates`、`fillna`、`to_parquet`
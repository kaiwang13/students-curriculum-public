#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学生自测入口 (Student self-check).

用你在 starter.py 里写好的实现，运行该题的官方测试，立刻给出 通过 / 未通过。

用法 Usage:
    python check.py <题包名>      检查一个练习      例：python check.py w1-oop
    python check.py w1-           检查名字以 w1- 开头的所有练习（某一周）
    python check.py --all         检查全部练习
    python check.py --progress    各周尝试进度概览（不跑测试，很快）
    python check.py --list        列出全部可检查的题包
    python check.py               显示本帮助

先在 assessments/autograder/<题包名>/starter.py 里把 raise NotImplementedError 换成你的实现，
再运行本命令。建议的完成顺序见 ORDER.md。

若报 缺依赖（如缺 torch / xgboost / statsmodels），按 env/setup.md 装好环境再试。
本命令用于自测；最终成绩以导师运行的完整评分器为准。
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKGS = ROOT / "assessments" / "autograder"
PASS, FAIL, TODO, DEP, TIME = "✅", "❌", "○", "⚠️", "⏱️"

# 每周练习包（由 scripts/build_dist.py 从 COURSE_ORDER 注入，供 --progress 使用）。
WEEKS = [
    ('基础', 'Python 工程', ['w1-oop', 'w1-dataclass', 'w1-cli', 'w1-decorators', 'w1-generators']),
    ('基础', 'NumPy 与向量化', ['w2-numpy', 'w2-broadcasting', 'w2-einsum', 'w2-ols', 'w2-softmax']),
    ('基础', 'pandas 与数据工程', ['w3-groupby', 'w3-merge', 'w3-io', 'w3-clean', 'p-w3-data-pipeline']),
    ('基础', '统计推断', ['w4-descriptive', 'w4-clt', 'w4-ttest', 'w4-ols-inference', 'w4-multiple-testing']),
    ('基础', '经典机器学习', ['w5-cv', 'w5-logreg', 'w5-pca', 'w5-metrics', 'w5-xgb']),
    ('基础', '深度学习入门 + nanograd L1–L3', ['w6-torch-basics', 'w6-nn-module', 'w6-training-loop', 'w6-vae', 'nanograd-l1', 'nanograd-l2', 'nanograd-l3', 'p-w6-vae']),
    ('通用技能', '张量自动微分 nanograd L4–L6', ['nanograd-l4', 'nanograd-l5', 'nanograd-l6']),
    ('通用技能', 'Transformer', ['w7-attention', 'w7-mha', 'w7-ft-transformer']),
    ('通用技能', '扩散模型与流匹配', ['w9-ddpm', 'w9-flow-matching', 'w9-sampler']),
    ('通用技能', '规模化训练', ['w10-ema', 'w10-checkpoint', 'w10-grad-accum']),
    ('通用技能', '研究工程（可复现性与指标）', ['cap-repro', 'cap-metrics']),
    ('通用知识', '关联检验', ['b7-maf', 'b7-assoc', 'b7-int', 'b7-gc']),
    ('通用知识', '群体结构与混合模型', ['b8-genopca', 'b8-ridge-lmm', 'b8-loco']),
]


def all_packages():
    return sorted(
        p.name for p in PKGS.iterdir()
        if p.is_dir() and (p / "test_hidden.py").exists() and (p / "starter.py").exists()
    )


def resolve(args):
    names = all_packages()
    if args == ["--all"]:
        return names
    picked = []
    for a in args:
        if a in names:
            picked.append(a)
        else:
            matches = [n for n in names if n.startswith(a)]
            if matches:
                picked.extend(matches)
            else:
                print("找不到题包: %s（用 --list 查看全部）" % a, file=sys.stderr)
    seen = set()
    return [n for n in picked if not (n in seen or seen.add(n))]


def _attempted(pkg):
    """一个题包是否已尝试：starter.py 里不再含 raise NotImplementedError。"""
    return "raise NotImplementedError" not in (
        (PKGS / pkg / "starter.py").read_text(encoding="utf-8"))


def progress():
    """快速概览：逐周统计已尝试 / 总数（只读 starter.py，不跑测试）。"""
    print("各周尝试进度（不运行测试，只看 starter.py 是否已动手）：\n")
    for wk, title, pkgs in WEEKS:
        checkable = [p for p in pkgs if (PKGS / p / "starter.py").exists()]
        if not checkable:
            continue
        done = sum(1 for p in checkable if _attempted(p))
        print("%-8s %s: 已尝试 %d/%d" % (wk, title, done, len(checkable)))
    print("\n提示：用 `python check.py <前缀>`（如 `python check.py w2-`）验证某周练习是否正确。")


def check_one(name):
    pkg = PKGS / name
    starter = (pkg / "starter.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy(pkg / "test_hidden.py", d / "test_hidden.py")
        shutil.copy(pkg / "starter.py", d / "solution.py")  # 你的 starter.py 充当 solution 模块
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "test_hidden.py", "-q",
                 "-p", "no:cacheprovider", "--tb=short"],
                cwd=str(d), capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            print("%s %s   超时（可能是死循环）" % (TIME, name))
            return "timeout"
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = [ln for ln in out.splitlines() if ln.strip()]
    summary = next((ln for ln in reversed(tail)
                    if any(k in ln for k in ("passed", "failed", "error"))), "")
    if proc.returncode == 0:
        print("%s %s   %s" % (PASS, name, summary))
        return "pass"
    # 缺依赖（collection 阶段 import 失败）优先于 ❌：不要把 traceback 甩给学生。
    if "ModuleNotFoundError" in out or "ImportError" in out:
        print("%s %s   缺依赖 —— 按 env/setup.md 安装后再试" % (DEP, name))
        return "dep"
    if "raise NotImplementedError" in starter and "NotImplementedError" in out:
        print("%s %s   未完成" % (TODO, name))
        print("    ↳ starter.py 里还有 NotImplementedError —— 把它们换成你的实现。")
        return "todo"
    print("%s %s   %s" % (FAIL, name, summary))
    for ln in out.strip().splitlines()[-30:]:
        print("    " + ln)
    return "fail"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print("共 %d 个可检查题包；用 `python check.py --list` 查看全部。" % len(all_packages()))
        return
    if args == ["--progress"]:
        progress()
        return
    if args == ["--list"]:
        print("\n".join(all_packages()))
        return
    targets = resolve(args)
    if not targets:
        sys.exit(1)
    tally = {"pass": 0, "fail": 0, "todo": 0, "dep": 0, "timeout": 0}
    for n in targets:
        tally[check_one(n)] += 1
    if len(targets) > 1:
        print("\n合计 %d 个：%s 通过 %d | %s 未通过 %d | %s 未完成 %d | %s 缺依赖 %d | %s 超时 %d"
              % (len(targets), PASS, tally["pass"], FAIL, tally["fail"],
                 TODO, tally["todo"], DEP, tally["dep"], TIME, tally["timeout"]))
    ok = all(tally[k] == 0 for k in ("fail", "todo", "dep", "timeout"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""学生自测入口 (Student self-check).

每一章的练习就在该章目录的 assignments/ 子文件夹里，与 lessons/ 并排。
在 assignments/<练习>/starter.py 里写你的实现，然后运行本命令自测。

用法:
    python check.py <练习名>        如：python check.py 02-broadcasting
    python check.py <章路径>        如：python check.py foundations/02-numpy   （整章）
    python check.py --all           检查全部
    python check.py --list          列出每章的练习
    python check.py --progress      每章的完成进度
"""
from __future__ import annotations
import re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CUR = ROOT / "curriculum"
PASS, FAIL, TODO, DEP, TIMED = "✅", "❌", "○", "⚠️", "⏱️"


def discover():
    items = []  # (chapter_relpath, folder, module, dir)
    for meta in sorted(CUR.rglob("assignments/*/meta.yaml")):
        d = meta.parent
        if not (d / "test_hidden.py").exists() or not (d / "starter.py").exists():
            continue
        m = re.search(r"^module:\s*(\S+)", meta.read_text(encoding="utf-8"), re.M)
        module = m.group(1) if m else d.name
        chapter = str(d.parent.parent.relative_to(CUR))
        items.append((chapter, d.name, module, d))
    return items


def resolve(args, items):
    if args == ["--all"]:
        return [it[3] for it in items]
    picked = []
    for a in args:
        exact = [it[3] for it in items if it[1] == a or it[2] == a]
        if exact:
            picked += exact; continue
        pref = [it[3] for it in items
                if it[1].startswith(a) or f"{it[0]}/{it[1]}".startswith(a) or it[0].startswith(a)]
        if pref:
            picked += pref
        else:
            print(f"找不到练习: {a}（用 --list 查看）", file=sys.stderr)
    seen = set()
    return [d for d in picked if not (d in seen or seen.add(d))]


def run_one(d):
    starter = (d / "starter.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        shutil.copy(d / "test_hidden.py", t / "test_hidden.py")
        shutil.copy(d / "starter.py", t / "solution.py")
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "test_hidden.py", "-q",
                                "-p", "no:cacheprovider", "--tb=short"],
                               cwd=str(t), capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"{TIMED} {d.name}   超时（可能死循环）"); return "timeout"
    out = (p.stdout or "") + (p.stderr or "")
    tail = [l for l in out.splitlines() if l.strip()]
    summ = next((l for l in reversed(tail) if any(k in l for k in ("passed", "failed", "error"))), "")
    if "ModuleNotFoundError" in out or "ImportError" in out:
        print(f"{DEP} {d.name}   缺依赖 — 按 env/setup.md 安装后再试"); return "dep"
    if p.returncode == 0:
        print(f"{PASS} {d.name}   {summ}"); return "pass"
    if "raise NotImplementedError" in starter and "NotImplementedError" in out:
        print(f"{TODO} {d.name}   尚未完成")
        print("    ↳ 在 starter.py 里把 NotImplementedError 换成你的实现。"); return "todo"
    print(f"{FAIL} {d.name}   {summ}")
    for l in out.strip().splitlines()[-30:]:
        print("    " + l)
    return "fail"


def main():
    from itertools import groupby
    args = sys.argv[1:]
    items = discover()
    if not args:
        print(__doc__); print(f"共 {len(items)} 个练习；`python check.py --list` 查看每章。"); return
    if args == ["--list"]:
        for ch, grp in groupby(items, key=lambda it: it[0]):
            print(f"\n{ch}/")
            for _, f, _, _ in grp:
                print(f"  {f}    (python check.py {f})")
        return
    if args == ["--progress"]:
        for ch, grp in groupby(items, key=lambda it: it[0]):
            g = list(grp)
            att = sum("raise NotImplementedError" not in (d / "starter.py").read_text(encoding="utf-8") for *_, d in g)
            print(f"{ch}: 已尝试 {att}/{len(g)}")
        return
    targets = resolve(args, items)
    if not targets:
        sys.exit(1)
    tally = {}
    for d in targets:
        r = run_one(d); tally[r] = tally.get(r, 0) + 1
    if len(targets) > 1:
        print("\n合计：" + " | ".join(f"{k}:{v}" for k, v in tally.items()))
    sys.exit(0 if not tally.get("fail") and not tally.get("todo") else 1)


if __name__ == "__main__":
    main()

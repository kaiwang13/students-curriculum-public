from __future__ import annotations
import argparse


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器"""
    raise NotImplementedError


def parse(argv: list[str]) -> dict:
    """解析命令行参数并返回字典"""
    raise NotImplementedError

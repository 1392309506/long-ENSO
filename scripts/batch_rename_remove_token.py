#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def build_new_stem(stem: str, token: str, ignore_case: bool = True) -> str:
    flags = re.IGNORECASE if ignore_case else 0
    new_stem = re.sub(re.escape(token), "", stem, flags=flags)
    new_stem = re.sub(r"[_-]{2,}", lambda m: m.group(0)[0], new_stem)
    new_stem = new_stem.strip("_- ")
    return new_stem


def iter_files(folder: Path, recursive: bool):
    if recursive:
        yield from (p for p in folder.rglob("*") if p.is_file())
    else:
        yield from (p for p in folder.iterdir() if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量删除文件名中的指定字符串（默认: monthly）"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="../data/era5/",
        help="目标文件夹，默认当前目录",
    )
    parser.add_argument(
        "--token",
        default="monthly",
        help="要删除的字符串，默认 monthly",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子目录",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行重命名；不加该参数时仅预览",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="区分大小写匹配 token",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="仅处理匹配的文件名（如 *.nc），默认 *",
    )
    parser.add_argument(
        "--regex-pattern",
        default=None,
        help="可选：用正则替换文件名 stem（不含后缀）",
    )
    parser.add_argument(
        "--regex-repl",
        default="",
        help="配合 --regex-pattern 使用的替换字符串",
    )

    args = parser.parse_args()
    folder = Path(args.folder).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"无效目录: {folder}")

    total = 0
    changed = 0
    skipped_conflict = 0

    for src in iter_files(folder, args.recursive):
        if not src.match(args.glob):
            continue
        total += 1
        if args.regex_pattern:
            flags = 0 if args.case_sensitive else re.IGNORECASE
            new_stem = re.sub(args.regex_pattern, args.regex_repl, src.stem, flags=flags)
            new_stem = re.sub(r"[_-]{2,}", lambda m: m.group(0)[0], new_stem)
            new_stem = new_stem.strip("_- ")
        else:
            new_stem = build_new_stem(
                src.stem,
                token=args.token,
                ignore_case=not args.case_sensitive,
            )

        if not new_stem or new_stem == src.stem:
            continue

        dst = src.with_name(new_stem + src.suffix)
        if dst == src:
            continue

        if dst.exists():
            skipped_conflict += 1
            print(f"[冲突跳过] {src.name} -> {dst.name}")
            continue

        changed += 1
        if args.apply:
            src.rename(dst)
            print(f"[已重命名] {src.name} -> {dst.name}")
        else:
            print(f"[预览] {src.name} -> {dst.name}")

    mode = "执行" if args.apply else "预览"
    print(
        f"\n完成（{mode}模式）：扫描 {total} 个文件，"
        f"可改名 {changed} 个，冲突跳过 {skipped_conflict} 个。"
    )


if __name__ == "__main__":
    main()

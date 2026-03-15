#!/usr/bin/env python3
"""
从 verl_demo.log 中提取推理正确的样本
用法: python extract_correct.py verl_demo.log [output.jsonl]
"""

import re
import json
import sys

def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def strip_prefix(line):
    return re.sub(r'^\(main_task pid=\d+\)\s*', '', line)

def clean_line(line):
    return strip_prefix(strip_ansi(line.rstrip('\n')))

def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    lines = [clean_line(l) for l in raw_lines]

    samples = []
    i = 0

    while i < len(lines):
        # 找样本起始行
        m = re.match(r'Target:\s*(\d+)\s*\|\s*Numbers:\s*\[([^\]]+)\]', lines[i])
        if not m:
            i += 1
            continue

        target = int(m.group(1))
        numbers = [int(x) for x in m.group(2).split()]

        # 向后找 "Correct equation:" 或 "No equation found" / "Invalid equation"
        # 同时收集这个样本块内的所有行
        block_start = i
        block_lines = []
        result_line = None
        j = i + 1

        while j < len(lines):
            l = lines[j]
            if re.match(r'Target:\s*\d+\s*\|\s*Numbers:', l):
                # 下一个样本开始了，说明当前样本没有结果行（异常情况）
                break
            block_lines.append(l)
            if l.startswith('Correct equation:'):
                result_line = l
                j += 1
                break
            if l.startswith('No equation found') or l.startswith('Invalid equation'):
                j += 1
                break
            j += 1

        i = j

        # 只保留正确的样本
        if result_line is None:
            continue

        # 从 block_lines 中提取 think 和 answer
        block_text = '\n'.join(block_lines)

        # 提取所有 <think>...</think> 块并拼接
        think_parts = re.findall(r'<think>(.*?)</think>', block_text, re.DOTALL)
        think = '\n'.join(part.strip() for part in think_parts if part.strip())
        # 兜底：处理没有闭合 </think> 的情况
        if not think:
            fallback = re.search(r'<think>(.*?)(?:<answer>|$)', block_text, re.DOTALL)
            if fallback:
                think = fallback.group(1).strip()

        # 提取 <answer>...</answer>（取最后一个，避免格式混乱）
        answer_matches = re.findall(r'<answer>(.*?)</answer>', block_text, re.DOTALL)
        answer = answer_matches[-1].strip() if answer_matches else ''

        # 提取 equation
        eq_match = re.match(r'Correct equation:\s*(.+)', result_line)
        equation = eq_match.group(1).strip() if eq_match else ''

        samples.append({
            'target': target,
            'numbers': numbers,
            'think': think,
            'answer': answer,
            'equation': equation,
        })

    return samples

def main():
    if len(sys.argv) < 2:
        print("用法: python extract_correct.py <log_file> [output.jsonl]")
        sys.exit(1)

    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else 'correct_samples.jsonl'

    samples = parse_log(log_path)

    with open(out_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"正确样本数: {len(samples)}")
    print(f"输出文件: {out_path}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 PDF 文件，提取标题、摘要和 bias 分类。

使用 openai_client 调用 API。
只处理未处理的 PDF。
保存结果为 jsonl, csv, xlsx。
"""

import os
import json
import glob
import pdfplumber
from openai_client import chat_completion
import pandas as pd
import jsonlines

def main():
    # 读取配置文件
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 处理记录文件
    processed_file = 'processed.json'
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed = json.load(f)
    else:
        processed = []

    # 查找所有 PDF 文件（假设在 pdfs 文件夹中，如果没有则在当前目录）
    pdf_dir = 'pdfs'
    if os.path.exists(pdf_dir):
        pdfs = glob.glob(os.path.join(pdf_dir, '*.pdf'))
    else:
        pdfs = glob.glob('*.pdf')

    results = []

    for pdf_path in pdfs:
        pdf_name = os.path.basename(pdf_path)
        if pdf_name in processed:
            print(f"Skipping already processed: {pdf_name}")
            continue

        print(f"Processing: {pdf_name}")

        # 提取 PDF 文本
        try:
            with pdfplumber.open(pdf_path) as pdf_file:
                text = ''
                for page in pdf_file.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
        except Exception as e:
            print(f"Failed to read PDF {pdf_name}: {e}")
            continue

        if not text.strip():
            print(f"No text extracted from {pdf_name}")
            continue

        # 构造完整 prompt
        full_prompt = config['prompt'] + "\n\nPDF 文本:\n" + text

        # 调用 API
        try:
            response = chat_completion(full_prompt,max_tokens=10000)  # 增加 max_tokens 以防摘要长
        except Exception as e:
            print(f"API call failed for {pdf_name}: {e}")
            continue

        # 检查响应是否为 JSON 格式的有效性
        if not response.strip():
            print(f"Empty response for {pdf_name}")
            continue

        # 去除 Markdown 代码块符号
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()

        # 尝试匹配第一个 { 和最后一个 }
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            response = response[start:end+1]
        else:
            print(f"Invalid JSON structure for {pdf_name}")
            print(f"Response: {response}")
            continue

        # 解析响应
        try:
            data = json.loads(response)
            data['filename'] = pdf_name
            # 确保 biases 是列表，如果是字符串则尝试解析
            if isinstance(data.get('biases'), str):
                data['biases'] = [b.strip() for b in data['biases'].split(',') if b.strip()]
            results.append(data)
            processed.append(pdf_name)
            print(f"Successfully processed: {pdf_name}")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for {pdf_name}: {e}")
            print(f"Response: {response}")
            continue

    # 保存处理记录
    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    if not results:
        print("No new results to save.")
        return

    # 合并新旧结果
    existing_results = []
    if os.path.exists('results.jsonl'):
        with jsonlines.open('results.jsonl', 'r') as reader:
            existing_results = list(reader)

    all_results = existing_results + results

    # 保存为 jsonl
    with jsonlines.open('results.jsonl', 'w') as writer:
        for item in all_results:
            writer.write(item)

    # 保存为 csv 和 xlsx
    df = pd.DataFrame(all_results)
    df.to_csv('results.csv', index=False, encoding='utf-8')
    df.to_excel('results.xlsx', index=False, engine='openpyxl')

    print("Results saved to results.jsonl, results.csv, results.xlsx")

if __name__ == '__main__':
    main()
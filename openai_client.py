#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量封装 OpenAI 协议的 chat/completions 调用。

主要接口：
  - chat_completion(prompt, model=None, temperature=0.0, max_tokens=2000)

该模块会自动从工作目录下的 .env 读取 `api_key`, `base_url`, `model`。
支持重试、指数退避、以及在必要时分片长 prompt。
返回模型的文本回复（字符串）。
"""

import os
import time
import json
from typing import Optional, List, Dict, Any
import requests
from dotenv import load_dotenv
try:
    from openai import OpenAI
    _HAS_OPENAI_SDK = True
except Exception:
    OpenAI = None
    _HAS_OPENAI_SDK = False

# load .env from workspace root if available
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV = os.path.join(HERE, '.env')
if os.path.exists(DEFAULT_ENV):
    load_dotenv(dotenv_path=DEFAULT_ENV)
else:
    load_dotenv()

API_KEY = os.getenv('api_key')
BASE_URL = os.getenv('base_url')
DEFAULT_MODEL = os.getenv('model') or 'gpt-4o'

if not API_KEY:
    raise RuntimeError('api_key not found in environment (.env)')
if not BASE_URL:
    raise RuntimeError('base_url not found in environment (.env)')

# normalize base url to not have trailing slash
BASE_URL = BASE_URL.rstrip('/')

DEFAULT_VERIFY = os.getenv('OPENAI_VERIFY_SSL', '1') != '0'

def _post_json(path: str, payload: Dict[str, Any], verify: bool = DEFAULT_VERIFY, timeout: int = 60) -> Dict[str, Any]:
    url = BASE_URL + path
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout, verify=verify)
    resp.raise_for_status()
    return resp.json()


def chat_completion(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    retries: int = 3,
    backoff_factor: float = 1.0,
    verify_ssl: Optional[bool] = None,
    truncate_chars: Optional[int] = 2000,
) -> str:
    """调用 /v1/chat/completions，返回完整文本响应。

    参数:
      - prompt: 用户级 prompt（字符串）
      - model: 覆盖 .env 中的默认模型
      - system: 可选 system 消息
      - temperature, max_tokens: 模型参数
      - retries: 失败重试次数
      - backoff_factor: 指数退避因子
      - verify_ssl: 是否验证 SSL 证书（None 则使用环境默认）
      - truncate_chars: 若 prompt 过长，将被截断为该长度

    返回:
      - 模型的文本回复（字符串）
    """
    model = model or DEFAULT_MODEL
    verify = DEFAULT_VERIFY if verify_ssl is None else verify_ssl

    # 安全截断
    send_prompt = prompt
    if truncate_chars and len(send_prompt) > truncate_chars:
        send_prompt = send_prompt[:truncate_chars] + "\n\n...<TRUNCATED>..."
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": send_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            # 优先使用 OpenAI SDK 客户端
            if _HAS_OPENAI_SDK and OpenAI is not None:
                try:
                    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=500
                    )
                    # SDK 返回结构通常在 resp.choices
                    choices = resp.get('choices') or []
                except Exception:
                    # 如果 SDK 调用失败，回退到 HTTP
                    choices = (_post_json('/v1/chat/completions', payload, verify=verify).get('choices') or [])
            else:
                choices = (_post_json('/v1/chat/completions', payload, verify=verify).get('choices') or [])

            texts: List[str] = []
            for ch in choices:
                # 支持多种返回结构
                if isinstance(ch, dict) and 'message' in ch and isinstance(ch['message'], dict):
                    cont = ch['message'].get('content') or ch['message'].get('text')
                else:
                    cont = ch.get('text') or ch.get('content') if isinstance(ch, dict) else None
                if cont:
                    texts.append(cont)

            if texts:
                return '\n'.join(texts).strip()
            # 如果没有提取到文本，但返回包含错误，则抛出
            if isinstance(choices, dict) and 'error' in choices:
                raise RuntimeError(f"LLM error: {choices['error']}")
            # 否则作为未知错误重试
            raise RuntimeError('No text in response')
        except Exception as e:
            last_err = e
            sleep_t = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_t)
    # all retries failed
    raise last_err if last_err else RuntimeError('Unknown error in chat_completion')


if __name__ == '__main__':
    # 简单示例：仅在本地运行时测试用
    test_prompt = '请简要列出高等数学中常见的主题（3-5 项）。'
    try:
        out = chat_completion(test_prompt)
        print('=== MODEL RESPONSE ===')
        print(out)
    except Exception as err:
        print('调用失败:', err)

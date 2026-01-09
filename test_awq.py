#!/usr/bin/env python3
"""测试 autoawq 是否正确安装"""
try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, AwqConfig
    print("SUCCESS: autoawq imported correctly")
except ImportError as e:
    print(f"FAILED: {e}")


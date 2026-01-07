"""
AWQ 量化脚本 - 完全模仿 eedi-mining-misconceptions 项目

用法：
    python awq_quantization.py --model_path /path/to/model --quant_path /path/to/output

注意：
    1. 需要安装 autoawq: pip install autoawq
    2. 如果 autoawq 有兼容性问题，可以直接使用 vLLM 运行时量化：
       python start_vllm_server.py --model /path/to/model --quantization awq
"""

import argparse

try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer, AwqConfig
    HAS_AWQ = True
except ImportError:
    HAS_AWQ = False
    print("⚠️ autoawq not installed. Install with: pip install autoawq")
    print("💡 Alternative: Use vLLM runtime quantization instead")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--quant_path", type=str, required=True, help="Path to save quantized model")
    parser.add_argument("--calib_data", type=str, default="pileval", help="Calibration dataset")
    parser.add_argument("--max_calib_seq_len", type=int, default=1024, help="Max calibration sequence length")
    args = parser.parse_args()
    
    if not HAS_AWQ:
        print("\n❌ Cannot run quantization without autoawq")
        print("\n💡 Alternative solutions:")
        print("   1. Install autoawq: pip install autoawq")
        print("   2. Use vLLM runtime quantization (no pre-processing needed):")
        print(f"      python start_vllm_server.py --model {args.model_path} --tensor-parallel-size 2")
        exit(1)
    
    model_path = args.model_path
    quant_path = args.quant_path
    calib_data = args.calib_data
    max_calib_seq_len = args.max_calib_seq_len

    # 量化配置 - 与 eedi-mining-misconceptions 完全一致
    quant_config = {"zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM"}

    print(f"\n{'='*60}")
    print(f"🔧 AWQ Quantization")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Output: {quant_path}")
    print(f"Calibration: {calib_data}")
    print(f"Max Seq Len: {max_calib_seq_len}")
    print(f"Config: {quant_config}")
    print(f"{'='*60}\n")

    # Load model
    print("📥 Loading model...")
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    print("🔧 Quantizing...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data, max_calib_seq_len=max_calib_seq_len)

    # Save
    print("💾 Saving...")
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    model.model.config.quantization_config = quantization_config
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f"\n{'='*60}")
    print(f"✅ Quantization complete!")
    print(f"{'='*60}")
    print(f"Output: {quant_path}")
    print(f"\n💡 Usage:")
    print(f"   python start_vllm_server.py --model {quant_path} --quantization awq --tensor-parallel-size 2")
    print(f"{'='*60}")

# Usage:
# python awq_quantization.py --model_path /home/user/models/deepseek-ai--DeepSeek-R1-Distill-Llama-70B --quant_path /home/user/models/DeepSeek-R1-70B-AWQ --calib_data pileval --max_calib_seq_len 1024

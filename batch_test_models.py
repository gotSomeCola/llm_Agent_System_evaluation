#!/usr/bin/env python3
"""
批量测试脚本：依次运行多个模型的评估
启动后可以去睡觉，脚本会自动处理所有测试
"""
import subprocess
import json
import time
import os
from datetime import datetime

# ========== 配置区域 ==========
# 要测试的模型列表
MODELS_TO_TEST = [

    "gemma3:12b",
    "llama3.3:70b",          # Meta Llama 3.3 - 70B 参数（可选）
    "llama3.1:70b",
    "llama3:8b",
    "llama3.1:8b",
    "gpt-oss:20b",
    "gpt-oss:120b",
    "hf.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:latest",
]

# 评估参数
EVAL_CONFIG = {
    "min_count": 0,
    "max_count": 1000,     # 测试 1000 个任务 ← 改这里
    "workers": 3,           # 4 个线程 ← 这个已经对了
    "k": 1,                 # pass@1 评估 ← 改这里
    "use_at_k": False       # 使用 pass@1 脚本（改为 False）← 改这里
}

# 输出目录
OUTPUT_DIR = "./results"

# ========== 主程序 ==========

def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_output_filename(model_name, timestamp):
    """生成输出文件名"""
    # 将模型名中的特殊字符替换为下划线
    safe_model_name = model_name.replace(":", "_").replace("-", "_")
    return os.path.join(OUTPUT_DIR, f"results_{safe_model_name}_{timestamp}.jsonl")

def run_evaluation(model_name, output_file, config):
    """运行单个模型的评估"""
    print(f"\n{'='*70}")
    print(f"🚀 启动评估: {model_name}")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出文件: {output_file}")
    print(f"配置: {config}")
    
    # 选择脚本
    script = "run_baseline_at_k.py" if config["use_at_k"] else "run_baseline.py"
    
    # 构建命令
    cmd = [
        "python", script,
        "--model_name", model_name,
        "--output_file", output_file,
        "--min_count", str(config["min_count"]),
        "--max_count", str(config["max_count"]),
        "--workers", str(config["workers"]),
    ]
    
    # 如果使用 pass@k，添加 k 参数
    if config["use_at_k"]:
        cmd.extend(["--k", str(config["k"])])
    
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        # 运行评估脚本
        print("⏳ 运行评估中...")
        result = subprocess.run(cmd, check=True)
        
        # 验证输出文件是否创建
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"\n✅ 评估成功!")
            print(f"JSONL 输出文件: {output_file}")
            print(f"文件大小: {file_size / 1024:.2f} KB")
            
            # 自动生成格式化报告
            print(f"\n⏳ 生成格式化报告中...")
            generate_report(model_name, output_file)
            
            return True
        else:
            print(f"\n⚠️ 警告: 输出文件未创建")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 评估失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        return False

def generate_report(model_name, jsonl_file):
    """为评估结果生成格式化报告"""
    try:
        cmd = [
            "python", "generate_results_report.py",
            "--input", jsonl_file,
        ]
        
        print(f"命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(f"✅ 报告生成成功!")
        print(f"输出文件:")
        
        # 提取生成的文件名
        base_path = os.path.splitext(jsonl_file)[0]
        output_dir = os.path.dirname(jsonl_file)
        
        report_files = {
            "CSV": f"{base_path}.csv",
            "TXT 总结": f"{base_path}_summary.txt",
            "分布图": f"{base_path}_distribution.png",
        }
        
        for file_type, file_path in report_files.items():
            if os.path.exists(file_path):
                print(f"  📄 {file_type}: {file_path}")
        
        # 显示 summary_all.csv 的路径
        summary_all = os.path.join(output_dir, "summary_all.csv")
        if os.path.exists(summary_all):
            print(f"  📊 累积对比: {summary_all}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 报告生成失败: {e}")
    except Exception as e:
        print(f"⚠️ 报告生成出错: {e}")

def main():
    """主程序"""
    print("\n" + "="*70)
    print("📊 LLM 代码生成评估 - 批量测试模式")
    print("="*70)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试模型数: {len(MODELS_TO_TEST)}")
    print(f"模型列表: {', '.join(MODELS_TO_TEST)}")
    print(f"评估模式: {'Pass@{} (多次尝试)'.format(EVAL_CONFIG['k']) if EVAL_CONFIG['use_at_k'] else 'Pass@1 (单次尝试)'}")
    print("="*70 + "\n")
    
    ensure_output_dir()
    
    # 记录时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_summary = {
        "start_time": datetime.now().isoformat(),
        "config": EVAL_CONFIG,
        "models": [],
        "end_time": None
    }
    
    # 依次运行每个模型
    for i, model_name in enumerate(MODELS_TO_TEST, 1):
        output_file = generate_output_filename(model_name, timestamp)
        
        print(f"\n[{i}/{len(MODELS_TO_TEST)}] 处理模型: {model_name}")
        
        # 运行评估
        success = run_evaluation(model_name, output_file, EVAL_CONFIG)
        
        # 记录结果
        model_result = {
            "model": model_name,
            "output_file": output_file,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        results_summary["models"].append(model_result)
        
        # 如果不是最后一个模型，等待一下
        if i < len(MODELS_TO_TEST):
            print(f"\n⏳ 等待 10 秒后继续下一个模型...")
            time.sleep(10)
    
    # 完成
    results_summary["end_time"] = datetime.now().isoformat()
    
    # 保存总结
    summary_file = os.path.join(OUTPUT_DIR, f"batch_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    # 打印最终总结
    print(f"\n\n" + "="*70)
    print("📋 批量测试完成!")
    print("="*70)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总结文件: {summary_file}\n")
    
    for i, model_result in enumerate(results_summary["models"], 1):
        status = "✅ 成功" if model_result["success"] else "❌ 失败"
        print(f"{i}. {model_result['model']}: {status}")
        print(f"   原始数据: {model_result['output_file']}")
        
        # 显示生成的报告文件
        base_path = os.path.splitext(model_result['output_file'])[0]
        report_files = {
            "CSV": f"{base_path}.csv",
            "TXT": f"{base_path}_summary.txt",
            "图表": f"{base_path}_distribution.png",
        }
        
        print(f"   格式化报告:")
        for file_type, file_path in report_files.items():
            if os.path.exists(file_path):
                print(f"      📄 {file_type}: {file_path}")
        print()
    
    # 显示累积对比文件
    summary_all_csv = os.path.join(OUTPUT_DIR, "summary_all.csv")
    if os.path.exists(summary_all_csv):
        print(f"📊 累积对比文件: {summary_all_csv}")
        print(f"   （包含所有模型运行的汇总数据，用于对比分析）\n")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

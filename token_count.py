from transformers import AutoTokenizer
from datasets import load_dataset
import torch

def count_specific_tokens(dataset_name, model_name):
    # 加载tokenizer
    print(f"正在加载{model_name}的tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载数据集
    print(f"正在加载{dataset_name}数据集...")
    dataset = load_dataset(dataset_name)
    
    # 为特定token找到对应的token ID
    newline_token = '\n'
    newline_token_id = 13
    
    # 对于EOS token，不同模型可能有不同的表示方式
    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    
    print(f"换行符token ID: {newline_token_id}, 表示为: {newline_token}")
    print(f"EOS token ID: {eos_token_id}, 表示为: {eos_token}")
    
    # 初始化计数器
    newline_count = 0
    eos_count = 0
    total_tokens = 0
    
    # 遍历数据集中的每个样本
    for split in dataset.keys():
        print(f"处理{split}分割...")
        
        # 假设代码在'content'或'code'字段中，根据实际情况调整
        code_field = 'content' if 'content' in dataset[split].column_names else 'code'
        if code_field not in dataset[split].column_names:
            print(f"警告: 在{split}分割中找不到代码字段，跳过")
            continue
        
        for i, sample in enumerate(dataset[split]):
            if i % 1000 == 0:
                print(f"已处理 {i} 个样本...")
            
            code = sample[code_field]
            tokens = tokenizer.encode(code, add_special_tokens=False)
            
            # 计数
            newline_count += tokens.count(newline_token_id)
            eos_count += tokens.count(eos_token_id)
            total_tokens += len(tokens)
    
    # 输出结果
    print("\n统计结果:")
    print(f"总token数: {total_tokens}")
    print(f"换行符(\\n)token数: {newline_count}, 占比: {newline_count/total_tokens*100:.2f}%")
    print(f"|<eos>| token数: {eos_count}, 占比: {eos_count/total_tokens*100:.2f}%")

if __name__ == "__main__":
    count_specific_tokens(
        "bigcode/the-stack-v2", 
        "codellama/CodeLlama-7b-hf"
    )
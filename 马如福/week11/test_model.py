"""
测试微调后的模型
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./output"  # 微调后的模型路径
# 如果还没有微调，可以使用本地原始模型路径进行测试
# MODEL_PATH = "./models/Qwen2-0.5B-Instruct"  # 本地模型路径


def test_model():
    """测试模型生成能力"""
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 测试问题
    test_contents = [
        # 测试案例1：社会新闻
        "近日，四川成都一市民在超市购物时，不慎将装有重要证件的背包遗落在收银台。超市工作人员发现后，通过背包内的名片联系到失主，最终将背包完好归还。失主为表感谢，特意赠送锦旗致谢。",
        # 测试案例2：科技新闻
        "我国科研团队近日宣布，在量子计算领域取得重大突破，成功研发出一款新型量子芯片，运算速度较传统芯片提升约100倍，有望为人工智能、密码学等领域带来革命性变化。",
        # 测试案例3：体育新闻
        "在昨晚结束的全国羽毛球锦标赛男单决赛中，年轻选手张某以2-1的比分战胜卫冕冠军李某，首次夺得全国冠军。赛后张某表示，将继续努力备战国际赛事。"
    ]
    
    print("\n开始测试...")
    print("=" * 50)
    
    for i, content in enumerate(test_contents, 1):
        # 构建输入（使用Qwen的对话格式）
        prompt = f"<|im_start|>user\n请为以下内容生成标题{content}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
            )
        
        # 解码
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取回答部分
        if "<|im_start|>assistant\n" in response:
            generated_title = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
        else:
            generated_title = "生成格式异常，未提取到标题！"
        
        print(f"测试案例: {i}")
        print(f"新闻内容: {content}")
        print(f"生成标题: {generated_title}")
        print("-" * 50)


if __name__ == "__main__":
    test_model()


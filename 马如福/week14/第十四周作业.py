"""
淘宝客服Agent示例 - 演示大语言模型的function call能力
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""

import os
import json
from openai import OpenAI


# ==================== 工具函数定义（淘宝客服核心能力） ====================
# 模拟淘宝客服常用的业务工具，实际场景会对接真实订单/物流/商品系统
def get_order_info(order_id: str):
    """
    根据订单号查询订单基本信息
    Args:
        order_id: 订单编号（如123456789）
    """
    # 模拟订单数据库
    orders = {
        "123456789": {
            "order_id": "123456789",
            "user_name": "张三",
            "product_id": "88888",
            "product_name": "新款纯棉T恤-白色-L码",
            "order_amount": 99.9,
            "pay_time": "2026-01-20 14:30:00",
            "order_status": "已付款待发货",
            "payment_method": "支付宝"
        },
        "987654321": {
            "order_id": "987654321",
            "user_name": "李四",
            "product_id": "99999",
            "product_name": "无线蓝牙耳机-标准版",
            "order_amount": 199.0,
            "pay_time": "2026-01-18 10:15:00",
            "order_status": "已发货",
            "payment_method": "微信支付"
        }
    }

    if order_id in orders:
        return json.dumps(orders[order_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "订单号不存在，请核对后重试"}, ensure_ascii=False)


def get_logistics_info(order_id: str):
    """
    根据订单号查询物流信息
    Args:
        order_id: 订单编号
    """
    # 模拟物流数据库
    logistics = {
        "123456789": {
            "order_id": "123456789",
            "logistics_company": "中通快递",
            "tracking_number": "7654321890123",
            "status": "待发货",
            "latest_update": "2026-01-20 14:30:00",
            "note": "仓库正在打包，预计24小时内发出"
        },
        "987654321": {
            "order_id": "987654321",
            "logistics_company": "顺丰速运",
            "tracking_number": "SF1234567890123",
            "status": "运输中",
            "latest_update": "2026-01-21 08:45:00",
            "location": "上海市→广州市",
            "note": "预计2026-01-22送达"
        }
    }

    if order_id in logistics:
        return json.dumps(logistics[order_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "暂无该订单的物流信息"}, ensure_ascii=False)


def get_product_stock(product_id: str):
    """
    根据商品ID查询库存和发货信息
    Args:
        product_id: 商品编号（如88888）
    """
    # 模拟商品库存数据库
    products = {
        "88888": {
            "product_id": "88888",
            "product_name": "新款纯棉T恤",
            "stock": 120,
            "stock_status": "有货",
            "delivery_place": "杭州市余杭区",
            "delivery_time": "付款后48小时内发货",
            "freight": "满59元包邮，不满59元运费8元"
        },
        "99999": {
            "product_id": "99999",
            "product_name": "无线蓝牙耳机",
            "stock": 0,
            "stock_status": "无货",
            "delivery_place": "深圳市南山区",
            "delivery_time": "补货后72小时内发货",
            "freight": "全场包邮"
        }
    }

    if product_id in products:
        return json.dumps(products[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "商品ID不存在"}, ensure_ascii=False)


def calculate_refund_amount(order_id: str, refund_reason: str):
    """
    根据订单号和退款原因计算可退款金额
    Args:
        order_id: 订单编号
        refund_reason: 退款原因（如"商品质量问题"、"七天无理由退货"、"拍错尺码"）
    """
    # 先获取订单基础信息
    order_info = json.loads(get_order_info(order_id))
    if "error" in order_info:
        return json.dumps(order_info, ensure_ascii=False)

    order_amount = order_info["order_amount"]
    product_id = order_info["product_id"]
    product_stock = json.loads(get_product_stock(product_id))

    # 退款规则（模拟）
    refund_amount = order_amount
    deduction = 0.0
    note = ""

    # 1. 质量问题：全额退款（含运费）
    if refund_reason == "商品质量问题":
        note = "商品质量问题支持全额退款，运费由商家承担"
    # 2. 七天无理由：包邮商品全额退，非包邮扣运费
    elif refund_reason == "七天无理由退货":
        if product_stock["freight"] == "全场包邮":
            note = "包邮商品七天无理由退货，全额退款"
        else:
            deduction = 8.0  # 扣运费8元
            refund_amount -= deduction
            note = f"非包邮商品七天无理由退货，扣除运费{deduction}元"
    # 3. 拍错尺码：扣运费（已发货）
    elif refund_reason == "拍错尺码":
        logistics_info = json.loads(get_logistics_info(order_id))
        if logistics_info["status"] == "已发货":
            deduction = 8.0
            refund_amount -= deduction
            note = f"商品已发货，拍错尺码退货需扣除运费{deduction}元"
        else:
            note = "商品未发货，拍错尺码可全额退款"
    else:
        refund_amount = order_amount
        note = "其他原因退款，按平台规则全额退款"

    result = {
        "order_id": order_id,
        "original_amount": order_amount,
        "deduction": deduction,
        "refund_amount": round(refund_amount, 2),
        "refund_reason": refund_reason,
        "note": note
    }
    return json.dumps(result, ensure_ascii=False)


def get_return_policy(product_id: str):
    """
    根据商品ID查询退换货政策
    Args:
        product_id: 商品编号
    """
    # 模拟退换货政策数据库
    policies = {
        "88888": {
            "product_id": "88888",
            "product_name": "新款纯棉T恤",
            "return_days": 7,
            "support_unsealed": True,
            "support_exchange": True,
            "note": "七天无理由退换，拆封后不影响二次销售可退，尺码不合适可换货（运费自理）"
        },
        "99999": {
            "product_id": "99999",
            "product_name": "无线蓝牙耳机",
            "return_days": 7,
            "support_unsealed": False,
            "support_exchange": True,
            "note": "七天无理由退换，拆封后不可退（电子类商品），非质量问题换货运费自理"
        }
    }

    if product_id in policies:
        return json.dumps(policies[product_id], ensure_ascii=False)
    else:
        return json.dumps({"error": "暂无该商品的退换货政策"}, ensure_ascii=False)


# ==================== 工具函数的JSON Schema定义 ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_info",
            "description": "根据订单号查询订单的基本信息，包括下单时间、商品名称、订单金额、订单状态等",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号，例如：123456789、987654321"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_logistics_info",
            "description": "根据订单号查询物流信息，包括快递公司、运单号、物流状态、最新更新时间等",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_stock",
            "description": "根据商品ID查询商品的库存状态、发货地、运费规则等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品编号，例如：88888、99999"
                    }
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund_amount",
            "description": "根据订单号和退款原因，计算可退款金额（含扣款规则）",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单编号"
                    },
                    "refund_reason": {
                        "type": "string",
                        "description": "退款原因，例如：商品质量问题、七天无理由退货、拍错尺码"
                    }
                },
                "required": ["order_id", "refund_reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_return_policy",
            "description": "根据商品ID查询商品的退换货政策，包括退换天数、是否支持拆封退货等",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "商品编号"
                    }
                },
                "required": ["product_id"]
            }
        }
    }
]

# ==================== Agent核心逻辑 ====================
available_functions = {
    "get_order_info": get_order_info,
    "get_logistics_info": get_logistics_info,
    "get_product_stock": get_product_stock,
    "calculate_refund_amount": calculate_refund_amount,
    "get_return_policy": get_return_policy
}


def run_taobao_agent(user_query: str, api_key: str = None, model: str = "qwen-plus"):
    """
    运行淘宝客服Agent，处理用户咨询
    Args:
        user_query: 用户输入的咨询问题
        api_key: API密钥（从环境变量读取或传入）
        model: 使用的大模型名称（通义千问系列）
    """
    # 初始化OpenAI兼容客户端（对接阿里云通义千问）
    client = OpenAI(
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": """你是一位专业的淘宝客服助手，负责解答用户的订单、物流、库存、退款、退换货等问题。
请根据用户的问题，调用合适的工具获取准确信息后，用友好、清晰的语言回复用户。"""
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    print("\n" + "=" * 60)
    print("【用户咨询】")
    print(user_query)
    print("=" * 60)

    # Agent循环（最多5轮工具调用）
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 第 {iteration} 轮Agent思考 ---")

        # 调用大模型
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        # 检查是否需要调用工具
        tool_calls = response_message.tool_calls

        if not tool_calls:
            # 无工具调用，返回最终回答
            print("\n【客服回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        # 执行工具调用
        print(f"\n【Agent决定调用 {len(tool_calls)} 个工具】")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称: {function_name}")
            print(f"工具参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行工具函数
            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                # 截断过长的返回结果，方便打印
                display_response = function_response[:200] + "..." if len(
                    function_response) > 200 else function_response
                print(f"工具返回: {display_response}")

                # 将工具结果加入对话历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，处理终止")
    return "抱歉，暂时无法解答你的问题，请稍后再试。"



if __name__ == "__main__":

    # 示例1：查询订单信息
    run_taobao_agent("帮我查一下订单号123456789的详细信息")

    # 示例2：查询物流信息
    run_taobao_agent("我的订单987654321到哪了？物流状态怎么样？")

    # 示例3：查询商品库存/运费
    run_taobao_agent("商品ID 88888还有货吗？发货地在哪？运费多少？")

    # 示例4：计算退款金额
    run_taobao_agent("我想退订单123456789的商品，原因是七天无理由退货，能退多少钱？")

    # 示例5：查询退换货政策
    run_taobao_agent("商品ID 99999支持拆封退货吗？退换货政策是什么？")

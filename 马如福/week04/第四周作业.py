#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

#待切分文本
sentence = "经常有意见分歧"


class AllCut:
    def all_cut(self, sentence, Dict):
        # 定义一个保存结果的数组
        result = []
        path = []
        """
        定义一个回溯方法
        1.确定参数
         1.1 要切分的字符串
         1.2 最终结果集
         1.3 每次遍历的结果集
         1.4 切割线即start_index
         1.5 条件满足集合
        2.确定终止条件
          2.1 切割的长度和字符串长度相等时即为停止
        3.单层搜索逻辑
          3.1 遍历是横向从左到右遍历
          3.2 进行切割，从start_index到i+1
          3.3 切割下来满足在Dict中的放到path中
          3.4 进行递归继续切分，知道切完整个字符串，然后就回溯
        """
        def backtracking(sentence, start_index, path, result, Dict):
            if start_index == len(sentence):
                result.append(path[:])
                return
            for i in range(start_index, len(sentence)):
                # 进行切割
                word = sentence[start_index: i + 1]
                if word in Dict:
                    path.append(word)
                    backtracking(sentence, i + 1, path, result, Dict)
                    path.pop()
        backtracking(sentence, 0, path, result, Dict)
        return result


if __name__ == "__main__":
    # 目标输出;顺序不重要
    target = [
     ['经常', '有意见', '分歧'],
     ['经常', '有意见', '分', '歧'],
     ['经常', '有', '意见', '分歧'],
     ['经常', '有', '意见', '分', '歧'],
     ['经常', '有', '意', '见分歧'],
     ['经常', '有', '意', '见', '分歧'],
     ['经常', '有', '意', '见', '分', '歧'],
     ['经', '常', '有意见', '分歧'],
     ['经', '常', '有意见', '分', '歧'],
     ['经', '常', '有', '意见', '分歧'],
     ['经', '常', '有', '意见', '分', '歧'],
     ['经', '常', '有', '意', '见分歧'],
     ['经', '常', '有', '意', '见', '分歧'],
     ['经', '常', '有', '意', '见', '分', '歧']
    ]
    cur_str = AllCut()
    res = cur_str.all_cut(sentence, Dict)
    # 测试输出的结果与给定目标结果是否一致
    for t in target:
        if len(res) != len(target):
            print("切分有误，重新切分")
        if t not in res:
            print("目标切分 %s 不在切分结果中" % t)
    print("切分正确！")



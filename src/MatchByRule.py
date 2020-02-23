"""
基于规则的分词技术一：
正向最大匹配算法

有一个词列表，基于这个列表对给定的中文文本进行分词。基本思想是：取列表中最长词的长度为步长。从第一个字符开始取最大步长字符串在此列表中查询匹配
如果匹配则向下继续，不匹配则去掉最后一个字符重复上述操作。
"""
class MM(object):
    def __init__(self, dictionary, dict_max_length):
        # 维护的词表
        self.dictionary = dictionary
        # 此表中词的最大长度
        self.dict_max_length = dict_max_length

    def cut(self, text: str) -> list:
        # 切分结果
        result = []
        current_index = 0
        while current_index < len(text):
            for i in range(self.dict_max_length, 0, -1):
                sub_text = text[current_index: current_index + i]
                if sub_text in self.dictionary:
                    # 如果在词表中找到，切分并进行下一轮匹配
                    result.append(sub_text)
                    current_index = current_index + i
                    break
        return result


"""
基于规则的分词技术二：
逆向最大匹配算法。

以字典中最长词长度为步长，从文本末尾开始进行匹配，若在字典中匹配到则进入下一个步长，若没有匹配到则去掉最前面的一个字继续匹配
，直到匹配或者只剩一个字，加入结果中，再向前移动一个步长，依次类推直到结束
"""
class RMM(object):
    def __init__(self, dictionary, maxLength):
        # 维护的词表
        self.dictionary = dictionary
        # 此表中词的最大长度
        self.maxLength = maxLength

    def cut(self, text: str) -> list:
        # 切分结果
        result = []
        textLength = len(text)
        currentIndex = textLength - 1
        while currentIndex >= 0:
            for length in range(self.maxLength, 0, -1):
                subText = text[currentIndex + 1 - length: currentIndex + 1]
                if subText in self.dictionary or length == 1:
                    # 如果在词表中找到，切分并进行下一轮匹配
                    result.append(subText)
                    currentIndex = currentIndex - length  # 移动指针
                    break
        # 反转列表为正向
        result.reverse()
        return result


"""
基于规则的分词技术三：
双向最大匹配算法。

进行正向和逆向最大匹配算法之后对结果进行比较，如果相同则返回任意一个，如果不同：
如果分词数量不同，返回分词数量较少的那个。
如果分词数量相同，返回单字数量较少的那个，如果单字数量相同，返回RMM的结果，因为RMM的认准率较高
"""
class BMM:
    def __init__(self, dictionary, maxLength):
        # 维护的词表
        self.dictionary = dictionary
        # 此表中词的最大长度
        self.maxLength = maxLength

    def cut(self, text: str) -> list:
        mmRes = MM(self.dictionary, self.maxLength).cut(text)
        rmmRes = RMM(self.dictionary, self.maxLength).cut(text)
        # 分词数量不同返回分词数较少的那个
        if not len(mmRes) == len(rmmRes):
            result = rmmRes if len(mmRes) > len(rmmRes) else mmRes
        else:
            # 完全一样返回任意一个，不一样返回单字较少的一个
            same = True
            mmSingleWordCount = 0
            rmmSingleWordCount = 0
            for i in range(len(mmRes)):
                if same and (not mmRes[i] == rmmRes[i]):
                    same = False
                if len(mmRes[i]) == 1:
                    mmSingleWordCount += 1
                if len(rmmRes[i]) == 1:
                    rmmSingleWordCount += 1
            # 完全一样返回任意一个
            if same:
                result = mmRes
            # 不一样返回单字较少的一个
            else:
                result = rmmRes if mmSingleWordCount >= rmmSingleWordCount else mmRes
        return result



if __name__ == '__main__':
    dic = ["研究", "研究生", "生命", "生", "命", "的", "起源"]
    textTest = "研究生命的起源"
    dict_max = 3

    # dic = ["南京", "南京市", "南京市长", "市长", "江", "大桥", "长江"]
    # textTest = "南京市长江大桥"
    # dict_max = 4

    mm = MM(dic, dict_max)
    print(mm.cut(textTest))

    rmm = RMM(dic, dict_max)
    print(rmm.cut(textTest))

    bmm = BMM(dic, dict_max)
    print(bmm.cut(textTest))



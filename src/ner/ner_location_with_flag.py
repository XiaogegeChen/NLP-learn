import time
import CRFPP

class NerLocationWithFlag:
    """
    增加词性作为一列特征
    """
    def handleCorpus(self):
        """
        处理语料库，对语料库中所有词进行状态标注，并抽取1/5作为测试集，剩余4/5作为训练集
        :return: void
        """
        def save(ch, ta, fl, file):
            """
            保存数据到指定的文件中
            :param fl: 词性
            :param ch: 字符集
            :param ta: 状态标签集
            :param file: 要写入的文件
            :return: void
            """
            for i in range(len(ch)):
                file.write(ch[i] + "\t" + fl[i] + "\t" + ta[i] + "\n")
            file.write("\n")

        with open("data/people_daily.txt", mode="r", encoding="utf8") as corpus,\
                open("data/trainingsetwithflag.txt", mode="w", encoding="utf8") as training,\
                open("data/testsetwithflag.txt", mode="w", encoding="utf8") as test:
            lineNum = 0
            for line in corpus:
                line = line.strip("\r\n\t")
                if line == "":
                    continue
                else:
                    words = line.split()[1:]
                    chars, tags, flags = self.handleLine(words)
                    if lineNum % 5 == 0:  # 20%为测试集
                        save(chars, tags, flags, test)
                    else:
                        save(chars, tags, flags, training)
                lineNum += 1

    def test(self, text):
        words = text.split()[1:]
        chars, tags, flags = self.handleLine(words)
        for i in range(len(chars)):
            print(chars[i] + "  ->  " + tags[i] + " -> " + flags[i])

    @staticmethod
    def handleLine(words):
        """
        处理一行中所有的词，拿到字符集和标签集
        :param words: 这一行的所有词
        :return: 字符集和标签集
        """
        def makeLabel(wo):
            """
            对地名做状态标签
            :param wo: 一个地表示名词语
            :return: 状态集合
            """
            res = []
            if len(wo) == 1:
                return ["S"]
            for m in range(len(wo)):
                if m == 0:
                    res.append("B")
                elif m == len(wo) - 1:
                    res.append("E")
                else:
                    res.append("M")
            return res

        def labelO(wo):
            return ["O"] * len(wo)

        def makeFlag(wo, fl):
            return [fl] * len(wo)

        chars = []  # 字符集
        tags = []  # 状态标签集
        flags = []  # 词性集
        builder = ""
        for word in words:
            word = word.strip()
            if builder == "":  # 不在构建合成词
                idx = word.find("[")
                if idx == -1:  # 不包含合成词
                    w, f = word.split("/")
                    # 做状态标注
                    if f == "ns":  # 地名
                        label = makeLabel(w)
                    else:
                        label = labelO(w)
                    # 做词性标注
                    flag = makeFlag(w, f)
                    chars.extend(w)
                    tags.extend(label)
                    flags.extend(flag)
                else:  # 包含合成词
                    w = word.split("/")[0][(idx + 1):]
                    builder += w
            else:  # 正在构建合成词
                idx = word.find("]")
                if idx == -1:  # 还没结束
                    builder += word.split("/")[0]
                else:  # 构建结束
                    w, f = word.split("/")
                    builder += w
                    f = word[idx + 1:]
                    # 做状态标注
                    if f == "ns":
                        # 地名
                        label = makeLabel(builder)
                    else:
                        label = labelO(builder)
                    # 做词性标注
                    flag = makeFlag(builder, f)
                    chars.extend(builder)
                    tags.extend(label)
                    flags.extend(flag)
                    builder = ""
        assert len(chars) == len(tags) and len(chars) == len(flags)
        return chars, tags, flags

    @staticmethod
    def calculatePRAndF1():
        with open("data/testresultwithflag.txt", mode="r", encoding="utf8") as f:
            allLocPre = 0  # 所有被模型识别为地名的数量
            locPreCorr = 0  # 被模型识别为地名且正确的数量
            allLocReal = 0  # 所有地名的数量
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                word, _, realFlag, preFlag = line.split()  # 字， 实际状态标注， 预测状态标注
                if preFlag != "O":
                    allLocPre += 1
                if realFlag != "O":
                    allLocReal += 1
                if realFlag == preFlag:
                    if not realFlag == "O":
                        locPreCorr += 1
            precision = locPreCorr * 1.0 / allLocPre  # 查准率
            recall = locPreCorr * 1.0 / allLocReal  # 召回率
            f1 = 2 * precision * recall / (precision + recall)  # 调和平均
            return precision, recall, f1

    @staticmethod
    def locationNER(text):
        tagger = CRFPP.Tagger("-m {0} -v 3 -n2".format("data/modelwithflag"))
        for c in text:
            tagger.add(c)
        res = []
        tagger.parse()
        builder = ""
        for i in range(tagger.size()):
            for j in range(tagger.xsize()):
                ch = tagger.x(i, j)
                tag = tagger.y2(i)
                if tag == "B":
                    builder = ch
                elif tag == "M":
                    builder += ch
                elif tag == "E":
                    builder += ch
                    res.append(builder)
                elif tag == "S":
                    builder = ch
                    res.append(builder)
        return res


if __name__ == '__main__':
    # nlwf = NerLocationWithFlag()
    # print("begin")
    # b = time.time()
    # nlwf.handleCorpus()
    # print("time consume -> " + str(time.time() - b) + "s")

    p, r, fOne = NerLocationWithFlag.calculatePRAndF1()
    print("precision -> " + str(p))
    print("recall -> " + str(r))
    print("f1 -> " + str(fOne))

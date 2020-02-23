import os
import pickle
import time

"""
基于统计的分词。通过使用隐含马尔可夫（HMM）模型实现。
HMM使用状态来表示一个字在一个词中的位置，如状态为[B, M, E, S]分别表示这个字在词语中词首、词中、词尾和单独成词。通过统计一定数量的语料
中的初始概率，发射概率、转移概率，再使用veterbi算法对传入的字符串逐个字符确定状态，从而得到一个最优路径及该路径的一个概率，进而达到分词
的目的。
首先使用特定的语料库训练模型，得到发射概率、转移概率、初始概率。然后使用veterbi算法逐个字符确定其状态，并记录经过的路径，最后选用概率最
大的路径作为最终状态，再根据状态得到分词结果
"""
class HMM:
    def __init__(self, trainingSetPath):
        # 分词语料存储路径
        self.trainingSetPath = trainingSetPath
        # 模型缓存路径
        self.modelPath = trainingSetPath + "_model"
        # 状态转移概率， 一个状态转移到另一个状态的概率
        self.transP = {}  # key是状态，value是一个字典，这个字典的key是状态，value是转移到这一个状态的概率
        # 发射概率，状态到词语的条件概率，（在某个状态下是某个字的概率）
        self.emitP = {}  # key是状态，value是一个字典，这个字典的key是具体的字，value是这个字被发射的概率
        # 状态初始概率
        self.startP = {}  # key是状态，value是这个状态作为初始状态的概率
        # 状态集合
        self.stateList = ["B", "M", "E", "S"]

    def loadModel(self):
        if os.path.exists(self.modelPath):
            # 已经训练好了，把结果读取进内存
            with open(self.modelPath, "rb") as f:
                self.transP = pickle.load(f)
                self.emitP = pickle.load(f)
                self.startP = pickle.load(f)
        else:
            # 训练模型
            self.trainModel()

    def trainModel(self):
        # 对一个词做状态标注
        def makeLabel(w):
            if len(w) == 1:
                res = ["S"]
            elif len(w) == 2:
                res = ["B", "E"]
            else:
                res = ["B"] + ["M"] * (len(w) - 2) + ["E"]
            return res

        countDic = {}  # 每个状态出现的次数，key 是状态，value为对应状态在训练集中出现的次数
        # 参数初始化
        for state in self.stateList:
            self.transP[state] = {s: 0.0 for s in self.stateList}
            self.emitP[state] = {}
            self.startP[state] = 0.0
            countDic[state] = 0
        lineNum = -1  # 训练集的行数
        wordSet = set()  # 字集合
        with open(self.trainingSetPath, encoding="utf8") as f:
            for line in f:
                lineNum = lineNum + 1
                line = line.strip()
                if not line:
                    continue
                wordList = [i for i in line if i != " "]  # 这一行的所有字和标点，去掉空格
                # 更新字集合
                wordSet |= set(wordList)
                lineList = line.split()  # 这一行的所有词语
                lineState = []  # 这一行每个字的状态
                # 状态标注
                for words in lineList:
                    lineState.extend(makeLabel(words))
                assert len(wordList) == len(lineState)
                # 更新初始概率，状态转移概率、发射概率
                for index, value in enumerate(lineState):
                    countDic[value] += 1
                    if index == 0:
                        # 更新状态初始概率
                        self.startP[value] += 1
                    else:
                        # 更新状态转移概率
                        self.transP[lineState[index - 1]][value] += 1
                        # 更新发射概率
                        self.emitP[value][wordList[index]] = self.emitP[value].get(wordList[index], 0) + 1
            # 将统计的初始状态次数转化为概率
            self.startP = {k: v * 1.0 / lineNum for k, v in self.startP.items()}
            # 将统计的状态转化次数转化为概率
            for k, v in self.transP.items():
                for k1, v1 in v.items():
                    self.transP[k][k1] = v1 * 1.0 / countDic[k]
            # 将统计的特定状态下某个字出现的次数转化成发射概率，需要加1平滑
            for k, v in self.emitP.items():
                for k1, v1 in v.items():
                    self.emitP[k][k1] = (v1 + 1) * 1.0 / countDic[k]
            # 缓存模型
            with open(self.modelPath, "wb") as modelFile:
                pickle.dump(self.transP, modelFile)
                pickle.dump(self.emitP, modelFile)
                pickle.dump(self.startP, modelFile)

    def viterbi(self, text, startP, transP, emitP):
        v = [{}]  # 递推的概率，是一个list，表示每个字是某个状态的概率，list的子项是字典，key是状态，value是这个状态的概率
        path = {}  # 路径，key是当前进度最后一个字的状态，value是从开始到当前进度key状态的最优路径
        # 确定初始概率
        for state in self.stateList:
            v[0][state] = startP[state] * emitP[state].get(text[0], 0)
            path[state] = [state]
        # 从第二个字开始递推，找到最优路径
        for t in range(1, len(text)):
            v.append({})
            newPath = {}  # 处理完这个字之后的新路径
            seen = False  # 这个字是否出现在发射概率中, 没出现的字一定会发射，单独成词
            for state in self.stateList:
                if text[t] in emitP[state].keys():
                    seen = True
                    break
            for y in self.stateList:  # y是下标t的字的状态
                # 因为使用的是二元语言模型，前面的一个字会影响后面的字，因此考虑上一个字的状态，找到从上一个字的状态转移到y状态的最大概率
                # 及状态，拿到状态转移路径
                maxP, state = -1, ""
                for y0 in self.stateList:  # y0是下标t-1的字的状态
                    p = v[t-1][y0] * transP[y0][y] * (emitP[y].get(text[t], 0) if seen else 1.0)
                    if p > maxP:
                        maxP, state = p, y0
                # 更新路径和递推概率
                newPath[y] = path[state] + [y]
                v[t][y] = maxP
            # 缓存路径
            path = newPath
        # 递推结束，从v中找到最后一个字最大的递推概率和相应的最后一个字的状态，再从path中取出相应状态的路径
        mState, mP = "", -1
        for state, p in v[len(text) - 1].items():
            if p > mP:
                mState, mP = state, p
        # 返回最大概率的状态路径及其概率
        return mP, path[mState]

    def cut(self, text: str):
        # if not os.path.exists(self.modelPath):
        #     self.trainModel()
        # 使用训练结果结合viterbi算法拿到最大概率的状态路径及概率值
        p, stateList = self.viterbi(text, self.startP, self.transP, self.emitP)
        begin, next = 0, 0
        for i, char in enumerate(text):
            state = stateList[i]
            if state == "B":
                begin = i
            elif state == "E":
                yield text[begin: i + 1]
                next = i + 1
            elif state == "S":
                yield char
                next = i + 1
        if next < len(text):
            yield text[next:]
        yield p


if __name__ == '__main__':
    hmm = HMM("data/trainingSet.txt")
    # start = time.time()
    # hmm.trainModel()
    # end = time.time()
    # print(hmm.startP)
    # print("time -> " + str(end - start))
    hmm.loadModel()
    res = hmm.cut("书中使用的语料库是人民日报的分词语料。测试一下：")
    print(str(list(res)))

import jieba
import os

"""
基于jieba库的高频词提取。先使用jieba分词，再统计词频并去掉停用词，找到次品最高的几个词
"""
class TF:
    def __init__(self, contentPath, stopWordsPath=""):
        # 初始内容
        self._originContent = ""
        # jieba分词结果， 每次遍历前都重新分词
        self._cutResult = []
        # 待提取的文本的路径
        self.contentPath = contentPath
        # 停用词路径
        self.stopWordsPath = stopWordsPath
        self._loadContent()

    def _loadContent(self):
        """
        把待切分的文本加载进内存
        :return: void
        """
        with open(self.contentPath, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                self._originContent += line

    def _cut(self):
        """
        分词
        :return: void
        """
        self._cutResult = jieba.cut(self._originContent)

    def getCut(self):
        """
        拿到分词结果
        :return: 分词结果
        """
        self._cut()
        return self._cutResult

    def getOrigin(self):
        """
        拿到原始文本
        :return: 原始文本
        """
        return self._originContent

    def getTF(self, topK=10):
        """
        获取高频词，没有过滤停用词
        :param topK: 前topK个高频词
        :return: 前topK个高频词
        """
        self._cut()
        wordsDic = {}
        for words in self._cutResult:
            wordsDic[words] = wordsDic.get(words, 0) + 1
        return sorted(wordsDic.items(), key=lambda x: x[1], reverse=True)[0: topK]

    def getTFWithStopWords(self, topK=10):
        """
        获取高频词，过滤了停用词
        :param topK: 前topK个高频词
        :return: 前topK个高频词
        """
        self._cut()
        stopWords = []
        if os.path.exists(self.stopWordsPath):
            with open(self.stopWordsPath, "r", encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    stopWords.append(line)
        wordsDic = {}
        for words in self._cutResult:
            wordsDic[words] = wordsDic.get(words, 0) + 1
        kv = sorted(wordsDic.items(), key=lambda x: x[1], reverse=True)
        res = []
        for k, v in kv:
            if len(res) == topK:
                break
            if k not in stopWords:
                res.append((k, v))
        return res


if __name__ == '__main__':
    tf = TF("data/news.txt", stopWordsPath="data/stopWords.txt")
    print(tf.getTF())
    print(tf.getTFWithStopWords())


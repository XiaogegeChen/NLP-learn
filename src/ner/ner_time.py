"""
命名实体识别之时间识别，通过分词及此行标注找到时间和数字词，再通过正则匹配解析时间并格式化为标准时间
"""
from jieba import posseg as psg
from datetime import timedelta, datetime
import re

ALL_NUM = re.compile(r"\d+$")
DAY_PATTERN = re.compile(r"[号|日]\d+$")
DIMENSION_PATTERN = re.compile(r"([0-9零一二两三四五六七八九十]+年)?([0-9零一二两三四五六七八九十]+月)?([0-9零一二两三四五六七八九十]+[号日])?([上中下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二两三四五六七八九十百]+分?)?([0-9零一二两三四五六七八九十百]+秒)?")
CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}
CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}

class TimeRecognition(object):
    def __init__(self):
        self._keyDayMap = {}
        self._loadKeyDayMap()

    def recognize(self, text: str):
        res = []
        if not text == "":
            # 先分词
            cutRes = list(psg.cut(text))
            print("---- cut res -----")
            for i in range(len(cutRes)):
                print(cutRes[i])
            # 拿到所有时间字符串
            allTimeStr = self._findAllTimeStr(cutRes)
            print("-------all time str ----")
            print(allTimeStr)
            # 筛选出合法的时间字符串, 转化为标准形式的时间
            for item in allTimeStr:
                if not self._checkTimeStr(item) is None:
                    parseTime = self._parseTimeStr(item)
                    if parseTime is not None:
                        res.append(parseTime)
        return res

    def _findAllTimeStr(self, cutResult):
        """
        拿到切分结果中所有表示时间的字符串
        :param cutResult: 分词结果
        :return: 所有表示时间的字符串
        """
        res = []
        subTimeStr = ""  # 用于拼接时间字符串
        for word, flag in cutResult:
            if word in self._keyDayMap.keys():
                if not subTimeStr == "":
                    # 停止拼接，加入结果中，置空等待下一次拼接
                    res.append(subTimeStr)
                    subTimeStr = ""
                # 指示代词转化成相应的时间描述
                t = datetime.today() + timedelta(days=self._keyDayMap.get(word, 0))
                subTimeStr = str(t.year) + "年" + str(t.month) + "月" + str(t.day) + "日"
            elif flag in ["m", "t"]:
                # 时间字符串进行拼接
                subTimeStr = word if subTimeStr == "" else subTimeStr + word
            else:
                # 如果正在拼接时间字符串，停止拼接，加入结果list，并置空等待下一次拼接
                if not subTimeStr == "":
                    res.append(subTimeStr)
                    subTimeStr = ""
        if not subTimeStr == "":
            res.append(subTimeStr)
        return res


    def _loadKeyDayMap(self):
        """
        加载时间指示代词
        :return: None
        """
        with open("data/keydays.txt", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                words = [word for word in line.split(" ") if word != ""]
                if len(words) == 2:
                    self._keyDayMap[words[0]] = int(words[1])

    def _checkTimeStr(self, timeStr):
        """
        校验时间的合法性，如果一个时间字符串不合法，返回None，合法则返回规范化之后的时间字符串
        :param timeStr: 时间字符串
        :return: 不合法返回None，合法则返回规范化之后的时间字符串
        """
        match = ALL_NUM.match(timeStr)
        if match:
            if len(timeStr) <= 6:
                return None
        newTimeStr = DAY_PATTERN.sub("日", timeStr)
        if newTimeStr == timeStr:
            return timeStr
        else:
            return self._checkTimeStr(newTimeStr)

    def _parseTimeStr(self, timeStr):
        if timeStr is None or len(timeStr) == 0:
            return None
        match = DIMENSION_PATTERN.match(timeStr)
        if match:
            if match.group(0) is not None:
                timeDic = {
                    "year": match.group(1),
                    "month": match.group(2),
                    "day": match.group(3),
                    "hour": match.group(5) if match.group(5) is not None else "00",
                    "minute": match.group(6) if match.group(6) is not None else "00",
                    "second": match.group(7) if match.group(7) is not None else "00"
                }
                newTimeDic = {}
                for item in timeDic.keys():
                    if timeDic[item] is not None and len(timeDic[item]) != 0:
                        if item == "year":
                            n = self._year2Num(timeDic[item][:-1])
                        else:
                            n = self._other2Num(timeDic[item][:-1])
                        if n is not None:
                            newTimeDic[item] = n
                targetDate = datetime.today().replace(**newTimeDic)
                # 一天内的时间段
                pm = match.group(4)
                if pm is not None:
                    if pm == "中午" or pm == "下午" or pm == "晚上":
                        hour = targetDate.hour
                        if hour < 12:
                            targetDate = targetDate.replace(hour=hour + 12)
                return targetDate.strftime("%Y-%m-%d %H:%M:%S")
        return None


    def _year2Num(self, year):
        res = ''
        for item in year:
            if item in CN_NUM.keys():
                res = res + str(CN_NUM[item])
            else:
                res = res + item
        m = re.match("\d+", res)
        if m:
            if len(m.group(0)) == 2:
                return int(datetime.today().year / 100) * 100 + int(m.group(0))
            else:
                return int(m.group(0))
        else:
            return None

    def _other2Num(self, src):
        if src == "":
            return None
        m = re.match("\d+", src)
        if m:
            return int(m.group(0))
        rsl = 0
        unit = 1
        for item in src[::-1]:
            if item in CN_UNIT.keys():
                unit = CN_UNIT[item]
            elif item in CN_NUM.keys():
                num = CN_NUM[item]
                rsl += num * unit
            else:
                return None
        if rsl < unit:
            rsl += unit
        return rsl


if __name__ == '__main__':

    # text1 = "我要从26号下午4点住到11月2号"
    # text2 = "我要住到明天下午三点"
    # text3 = "预定28号的房间"
    text4 = "06秒"
    tr = TimeRecognition()
    # print(tr.recognize(text1))
    # print(tr.recognize(text2))
    # print(tr.recognize(text3))
    print(tr.recognize(text4))

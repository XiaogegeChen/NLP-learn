from jieba import posseg as psg
from datetime import datetime, timedelta
import re

if __name__ == '__main__':
    r = psg.cut("hello 你好呀！中国， 北京天安门")
    for item in r:
        print(item)





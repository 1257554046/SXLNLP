#week3作业
import re
import time
#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"
#count = 0
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):

    #TODO
    #maxlen = len(max(Dict.keys()))
    words = []
    def dfs(start, path):
        #global count
        if start == len(sentence):
            if path:
                words.append(path)
            print('stop')
            return
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start : end]
            print('start:', start)
            print('end:',end)

            if word in Dict:
                print(path+[word])
                dfs(end, path + [word])
                #count = count+1
                #print(count)

    dfs(0,[])
    return words

two_d_list = all_cut(sentence, Dict)
for inner_list in two_d_list:
    print(inner_list)


#目标输出;顺序不重要
'''
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
'''



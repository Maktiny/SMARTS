'''
Author:  lixu
Date: 2021-06-28 14:00:18
LastEditTime: 2021-06-28 19:41:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /SMARTS/multi_merge/starter_kit/agg_merge/ppo/plot.py
'''

#-*- coding: utf-8-*-
import matplotlib.pyplot as plt
import re
def getlog(logfilepath):
    f = open(logfilepath,"r",encoding="utf-8")
    line = f.readline()
    exposide = []
    reward = []

    while line:
        number = re.compile(r'-?\d+')
        tem = number.findall(line)
        
        exposide.append(int(tem[0]))
        reward.append(float(tem[1]))
        line = f.readline()
    f.close()
    return exposide, reward

def plotimg(exposide,reward):
    plt.plot(exposide,reward)
    plt.xlabel("exposide")
    plt.ylabel("reward")
    plt.title("aggressive merge accumulation reward caver")
    plt.grid(True)
    plt.savefig("/home/carserver2/SMARTS/multi_merge/temp/data/2021-06-30-19-21-26/"+"reward.jpg")
    plt.show()

if __name__=="__main__":
    path = "/home/carserver2/SMARTS/multi_merge/temp/data/2021-06-30-19-21-26/log.txt"
    exposide, reward = getlog(path)
    print(path)
    plotimg(exposide, reward)



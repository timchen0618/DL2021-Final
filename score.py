import sys
import os
# age:	37.433223724365234	, image:	28_0_1_20170113133358713.jpg.chip.jpg
def cal_score(filename):
    lines = [l.strip('\n') for l in open(filename)]
    mae = []
    for l in lines:
        pred = float(l.split('\t')[1])
        gold = float(l.split('\t')[3].split('/')[-1].split('_')[0])
        mae.append(abs(pred-gold))

    print(sum(mae), len(mae))
    print(sum(mae)/len(mae))
    return (sum(mae)/len(mae))

if __name__ == '__main__':
    cal_score(sys.argv[1])

import shutil
import sys
import os
import random
path=sys.argv[1]
files=os.listdir(path)
images_names=[[], [], [], [], []]
dirs=['white', 'black', 'asian', 'indian', 'others']

for l in files:
    if len(l.split('_')) > 2:
        if l.split('_')[2] and not (l[0] == '.'):
            try:
                rid = int(l.split('_')[2])
                images_names[rid].append(l)
                shutil.move(os.path.join(path, l), os.path.join(path, dirs[rid], l))
            except:
                print(l)
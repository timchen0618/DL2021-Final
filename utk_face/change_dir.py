import shutil
import sys
import os
import random
# path=sys.argv[1]
# files=os.listdir(path)
# images_names=[[], [], [], [], []]
# dirs=['white', 'black', 'asian', 'indian', 'others']

# for l in files:
#     if len(l.split('_')) > 2:
#         if l.split('_')[2] and not (l[0] == '.'):
#             try:
#                 rid = int(l.split('_')[2])
#                 images_names[rid].append(l)
#                 shutil.move(os.path.join(path, l), os.path.join(path, dirs[rid], l))
#             except:
#                 print(l)


# part1/white
# part2/black
# part3/asian, ...
# create train.py


dirs=['white', 'black', 'asian', 'indian', 'others']
# for l in dirs:
#     new_dirs = list(dirs)
#     new_dirs.remove(l)
    # print(new_dirs)
    # print(dirs)
    # exit(0)
for l in dirs:
    new_dirs = [l]

    images = {k:[] for k in new_dirs}
    outdirs = ['part1', 'part2', 'part3']
    for out in outdirs:
        for in_ in new_dirs:
            files = os.listdir(os.path.join(out, in_))
            images[in_].extend([(os.path.join(out, in_), f) for f in files])
        
    for in_ in new_dirs:
        print(in_, len(images[in_]))
        random.shuffle(images[in_])

    image_split = {'train':[], 'test':[]}

    for race in new_dirs:
        len_train = int(len(images[race])*0.9)
        train = images[race][:len_train]
        test = images[race][len_train:]

        image_split['train'].extend(train)
        image_split['test'].extend(test)

    for k, v in image_split.items():
        fw = open('only_%s_%s.csv'%(l, k), 'w')
        fw.write('filename,age\n')
        for image in v:
            fw.write(os.path.join(image[0], image[1]))
            fw.write(',')
            fw.write(image[1].split('_')[0])
            fw.write('\n')
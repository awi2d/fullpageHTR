import os
import random

import Dataloader
import main


def test_linepointenc():
    points = [((0, 1), (2, 3), 4), ((5, 6), (7, 8), 9)]
    tp = [(0, 1), (2, 3), (4, 5)]
    x = 8
    y = 10
    assert Dataloader.dense2linepoints(Dataloader.linepoint2dense(points, x, y, len(points)), x, y) == points
    assert Dataloader.sparse2linepoints(Dataloader.linepoint2sparse(points, x, y, len(points)), x, y) == points
    assert Dataloader.dense2points(Dataloader.points2dense(tp, 32), 32) == tp

def op(n, args):
    #print("op(", n, ", ", args)
    if n < 1:
        return "NaN"
    if n == 1:
        return sum(args)
    if n == 2:
        r = 1
        for x in args:
            r *= x
        return r
    if len(args) == 1:
        return args[0]
    return op(n, [op(n-1, [args[0]]*args[1])] + args[2:])

#for i in range(10):
#    print("op(", i, ", [2, 2]) = ", op(i, [2, 2]))


def getDirSize(dir, size):
    if os.path.isfile(dir):
        size[dir] = os.path.getsize(dir)
        return size
    if os.path.isdir(dir):
        subfiles = []
        try:
            subfiles = os.scandir(dir)
        except:
            print("cant read subfiles of ", dir)
        selfSize = 0
        for subf in [f.path for f in subfiles]:
            size = getDirSize(subf, size)
            selfSize += size[subf]
        size[dir] = selfSize
        return size
    print("file ", dir, " is no file nor dir")
    size[dir] = 0
    return size


def printDirSize():
    size = getDirSize("C:\\", {})
    print("size = ", size)
    with open("C:/Users/Idefix/Documents/dirsizes.txt", 'w') as f:
        keys = list(size.keys())
        try:
            keys.sort(key=lambda x: str(x).count('\\'))
            for k in keys:
                print(str(k)+" - ", file=f)
        except:
            print("cant print: unprintable")

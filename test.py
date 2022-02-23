import Dataloader

def test_linepointenc():
    points = [((0, 1), (2, 3), 4), ((5, 6), (7, 8), 9)]
    x = 8
    y = 10
    assert Dataloader.dense2linepoints(Dataloader.linepoint2dense(points, x, y, len(points)), x, y) == points
    assert Dataloader.sparse2linepoints(Dataloader.linepoint2sparse(points, x, y, len(points)), x, y) == points

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

for i in range(10):
    print("op(", i, ", [2, 2]) = ", op(i, [2, 2]))

import Dataloader


def test_linepointenc():
    points = [((0, 1), (2, 3), 4), ((5, 6), (7, 8), 9)]
    x = 8
    y = 10
    assert Dataloader.dense2points(Dataloader.point2dense(points, x, y, len(points)), x, y) == points
    assert Dataloader.sparse2points(Dataloader.point2sparse(points, x, y, len(points)), x, y) == points

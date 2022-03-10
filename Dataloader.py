import cv2
import numpy as np
import tensorflow as tf
import random
#import tensorflow_datasets as tf_ds
# TODO https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# TODO tf.keras.utils.image_dataset_from_directory nutzen
# 125 GB RAM auf server (gesamt)

line_point = ((int, int), (int, int), int)  # (startpoint of line, endpoint of line, height)

data_dir = "../SimpleHTR/data/trainingDataset/"
#data_dir = "C:/Users/Idefix/PycharmProjects/SimpleHTR/data/"  # The dirctory that is mapped to not be in the docker
iam_dir = data_dir+"iam/"  # the unchanged iam dataset
dataset_dir = data_dir+"generated/"  # directoy for storing generated data
models_dir = data_dir+"models/"  # directoy for storing trained models


class goldlabel_types:
    text = 0
    linepositions = 1
    number_of_lines = 2


class goldlabel_encodings:
    onehot = 0
    dense = 1


class img_types:
    word = 0
    line = 1
    paragraph = 2


class dataset_names:
    iam = 0


def num2sparse(num, max):
    r = [0]*max
    r[num] = 1
    return r
def sparse2num(sparse):
    r = 0
    for i in range(len(sparse)):
        if sparse[i]>sparse[r]:
            r = i
    return r


def point2spares(point: (int, int)):
    r = [0]*(32*2)
    r[point[0]] = 1
    r[point[1]+32] = 1
    return r
def sparse2point(point: [float]):
    x = 0
    y = 0
    for i in range(32):
        if point[i] > point[x]:
            x = i
        if point[i+32] > point[y+32]:
            y = i
    return (x, y)

def points2dense(points: [(int, int)], max_x=32, max_y=32) -> [float]:
    return [points[int(i/2)][i%2]/max_x for i in range(len(points)*2)]

def dense2points(points: [float], max_x=32, max_y=32) -> [(int, int)]:
    return [(int(points[2*i]*max_x), int(points[2*i+1]*max_x)) for i in range(int(len(points)/2))]


# <debug functions>


def downscale(img, goldlabel: [line_point], x: int, y: int):
    """
    reduces the resolution of the image
    :param img:
    an opencv-img, that means numpy-array
    :param goldlabel:
    the goldlabel of image, unencoded
    :param x:
    the factor by witch the width of the image is scaled
    :param y:
    the factor by witch the hight of the image is scaled
    :return:
    an image of size (input_size[0]/x, input_size[1]/y)
    """
    img = cv2.resize(img, dsize=None, fx=1/x, fy=1/y)
    goldlabel = [((int(x1/x), int(y1/y)), (int(x2/x), int(y2/y)), int(h/y)) for ((x1, y1), (x2, y2), h) in goldlabel]
    return np.array(img, dtype="uint8"), goldlabel


def sample_linepoint(img, goldlabel: [line_point], upperleftcorner: (int, int), sampesize: (int, int)):
    """
    samples an area out of the image
    :param img:
    an oopencv-img, that means numpy-array
    :param upperleftcorner:
    the upper left corner of the return image in the input image
    :param sampesize:
    the size of the return image
    :return:
    an image that shows an area out of th input image
    """
    ulx, uly = upperleftcorner
    drx = ulx+sampesize[0]
    dry = uly+sampesize[1]
    goldlabel = [((x1-ulx, y1-uly), (x2-ulx, y2-uly), h) for ((x1, y1), (x2, y2), h) in goldlabel if ulx < min(x1, x2) and uly < min(y1, y2) and max(x1, x2) < drx and max(y1, y2) < dry]
    if len(goldlabel) == 0:
        goldlabel = [((0, 0), (0, 0), 0)]
    return np.array(img[uly:dry, ulx:drx], dtype="uint8"), goldlabel

def sample_point(img, goldlabel: [line_point], upperleftcorner: (int, int), sampesize: (int, int)):
    """
    samples an area out of the image
    :param img:
    an oopencv-img, that means numpy-array
    :param upperleftcorner:
    the upper left corner of the return image in the input image
    :param sampesize:
    the size of the return image
    :return:
    an image that shows an area out of th input image
    """
    ulx, uly = upperleftcorner
    drx = ulx+sampesize[0]
    dry = uly+sampesize[1]
    goldlabel = [(x-ulx, y-uly) for (x, y) in goldlabel if ulx < x and uly < y and x < drx and y < dry]
    return np.array(img[uly:dry, ulx:drx], dtype="uint8"), goldlabel

def get_testdata(enc = goldlabel_encodings.dense):
    """
    :param enc:
    element of goldlabel_encodings
    :return:
    [(img, goldlabel)]
    """
    #TODO testdaten die näher an linien sind generieren.
    if enc == goldlabel_encodings.dense:
        encfunc = points2dense
    elif enc == goldlabel_encodings.onehot:
        encfunc = point2spares
    else:
        print("Dataloader.get_testdata: invalid encoding: ", enc)
        return None
    size = 32
    r = []
    for i in range(size**2):
        pos = (int(i/size), i%size)
        #cv2.circle(img, center, radius, color)

        img = np.full((size, size), 255, dtype='uint8')
        cv2.circle(img, pos, 5, 0, thickness=5)
        goldlabel = encfunc(pos)
        r.append((img, goldlabel))
        #r[i] = (cv2.circle(r[i][0], pos, 2, 0), (pos[0]/size, pos[1]/size))  # dense encoding
    random.shuffle(r)
    return r


def getType(x):
    name = type(x).__name__
    if name == 'list':
        return '['+str(len(x))+":"+getType(x[0])+']'  # TODO assumes all element of the list have the same type
    if name == 'tuple':
        r = '<'+str(len(x))+":"
        for i in x:
            r += getType(i)+'; '
        return r[:-2] + '>'
    if name == 'ndarray':
        return 'ndarray('+str(x.shape)+': '+(getType(x[0]) if len(x) > 0 else "Nix")+')'
    if name == 'BatchDataset':
        return str(name)+" : "+str(len(x))  # TODO
    return name


def test_tfds():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(1626, 1132),  # = (img_height, img_width)
        batch_size=128,
        labels=None)  # TODO labels = None
    # TODO zweites Dataset mit goldlabels
    print("train_ds = ", train_ds)
    print("type(train_ds) = ", getType(train_ds))
    print("class_names = ", train_ds.class_names)
    print("train_ds.element_spec= ", train_ds.element_spec)
    print("train_ds length: ", len(train_ds))  # train_ds length: 2, anzahl der Bilder: 199
    for elem in train_ds:
        print("elem = ", getType(elem))  # elem =  EagerTensor
        break
    goldlabel = []
    with open(dataset_dir+"/dir-gl.txt", 'r') as f:
        for l in f.readlines():
            point = l.split('-(')[1].replace(")\n", "").split(", ")  # TODO may only work for (float, float) tupels
            point = (float(point[0]), float(point[1]))
            goldlabel.append(point)
    goldlabel = np.array(goldlabel)
    return train_ds, goldlabel


r = 1
def random_choice(list):
    # looks kinda random, all results are equaly likely, but is always the same and didnt need any additional packages.
    global r
    r = (r+97)%251  # may not work correktly if len(list)%97=0 for multiple cals
    return list[r%len(list)]

#</debug functions>


def load_img(filename):
    """
    :param filename:
    the path to an single image
    :return:
    an grayscale image.
    """
    img = cv2.imread(filename)
    if img.shape[2] == 3:  # convert from rgb to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # uint8


def txt2sparse(txt, alphabet, y_size):
    a = np.array([alphabet[c] for c in txt])
    b = np.zeros((a.size, len(alphabet)))
    b[np.arange(a.size),a] = 1
    b = np.pad(b, ((0,0), (0, y_size-b.shape[0])))  # TODO
    return np.array(b)


def linepoint2sparse(points: [line_point], max_x: int, max_y: int, y_size: int) -> [float]:
    assert max([max(point[0][0], point[1][0]) for point in points]) < max_x
    assert max([max(point[0][1], point[1][1], point[2]) for point in points]) < max_y
    assert y_size >= len(points)
    point_length = 2 * max_x + 3 * max_y  # number of bits needed to encode one point
    r = [0]*point_length*y_size
    #r[s*point_length+0:s*point_length+max_x] = one-hot encoding von points[s][0][0]
    for i in range(len(points)):
        ((xs, ys), (xe, ye), h) = points[i]
        offset = point_length*i
        r[offset+int(xs)] = 1
        r[offset + max_x + int(ys)] = 1
        r[offset + max_x + max_y + int(xe)] = 1
        r[offset + 2 * max_x + max_y + int(ye)] = 1
        r[offset + 2 * max_x + 2 * max_y + int(h)] = 1
    return np.array(r)


def sparse2linepoints(points: [float], max_x: int, max_y: int) -> [line_point]:
    point_length = 2*max_x+3*max_y
    points = [points[i:i+point_length] for i in range(0, len(points), point_length)]
    r = []
    for point in points:
        # point = [0 or 1]*(3h+2w)
        #print("point = ", point)
        xs = 0
        ys = 0
        xe = 0
        ye = 0
        h = 0
        for i in range(max_x):
            if point[i] > point[xs]:  # and point[i] > 0.1:
                xs = i
            if point[i+max_x+max_y] > point[xe+max_x+max_y]:  # and point[i+max_x+max_y] > 0.1:
                xe = i
        for i in range(max_y):
            if point[i+max_x] > point[ys+max_x]:
                ys = i
            if point[i+2*max_x+max_y] > point[ye+2*max_x+max_y]:
                ye = i
            if point[i+2*max_x+2*max_y] > point[h+2*max_x+2*max_y]:
                h = i
        r.append(((xs, ys), (xe, ye), h))
    return r


def linepoint2dense(points: [line_point], max_x: int, max_y: int, y_size: int) -> [float]:
    assert max([max(point[0][0], point[1][0]) for point in points]) <= max_x
    assert max([max(point[0][1], point[1][1], point[2]) for point in points]) <= max_y
    assert y_size >= len(points)

    dp = [(xs/max_x, ys/max_y, xe/max_x, ye/max_y, h/max_y) for ((xs, ys), (xe, ye), h) in points]
    dp = list(sum(dp, ()))  # flattens the list. https://stackoverflow.com/questions/10632839/transform-list-of-tuples-into-a-flat-list-or-a-matrix/35228431
    return np.array(dp+[0]*(5*y_size-len(dp)))


def dense2linepoints(points: [float], max_x: int, max_y: int) -> [line_point]:
    assert len(points)%5 == 0
    return [((int(points[i]*max_x), int(points[i+1]*max_y)), (int(points[i+2]*max_x), int(points[i+3]*max_y)), int(points[i+4]*max_y)) for i in range(0, len(points), 5)]


def fitsize(img, gl, w, h):
    if img.shape[0] >= h:
        img, gl = sample_linepoint(img, gl, (0, 0), (img.shape[0], h-1))
    if img.shape[1] >= w:
        img, gl = sample_linepoint(img, gl, (0, 0), (w-1, img.shape[1]))
    #print("fitsize: (h, w) = ", (h, w), "shape = ", img.shape)
    return np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), gl


def encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=None, y_size=1):
    """
    :param data:
    data of type [(image, goldlabel)]
    :param goldlabel_type:
    type of goldlabel in Dataset.gl_type
    currently text or lineposition
    :param goldlabel_encoding:
    encoding of goldlabel in Dataset.gl_encoding
    :param size:
    (width: int, height: int) or the other way round, size all images in data will have.
    that means they either get padded are cliped.
    :param y_size:
    size all goldlabel_encodings in data will have
    :return:
    [(image, goldlabel)] like input, but the goldlabel is encoded and image and goldlabel are padded to size
    """

    if size is None:
        h = int(max([img.shape[0] for (img, gl) in data])/2)  # konstants should always be the same as in rescaling. This will obviusly break with goldlabel_type == text
        w = int(max([img.shape[1] for (img, gl) in data])/4)
    else:
        h = size[0]
        w = size[1]
    print("Dataloader.encode_and_pad: w, h = ", h, ", ", w)
    if goldlabel_type == goldlabel_types.text:
        # get alphabet used
        alphabet_int2char = list(set([c for c in ''.join([txt for (img_path, txt) in data])]))
        alphabet_char2int = {}
        for i in range(len(alphabet_int2char)):
            alphabet_char2int[alphabet_int2char[i]] = i

        if goldlabel_encoding == goldlabel_encodings.dense:
            print("dense text encoding is currently unsupported, return text in UTF-8 as goldlabel instead")
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), txt) for (img, txt) in data], alphabet_int2char
        elif goldlabel_encoding == goldlabel_encodings.onehot:  # text in one-hot
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), txt2sparse(txt, alphabet_char2int, y_size)) for (img, txt) in data], alphabet_int2char
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == goldlabel_types.linepositions:
        data = [downscale(img, points, 4, 2) for (img, points) in data]  # TODO should be optional
        #h = int(h/2)
        #w = int(w/4)
        data = [fitsize(img, gl, w, h) for (img, gl) in data]
        if goldlabel_encoding == goldlabel_encodings.dense:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), linepoint2dense(point, max_x=w, max_y=h, y_size=y_size)) for (img, point) in data]
        elif goldlabel_encoding == goldlabel_encodings.onehot:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), linepoint2sparse(point, w, h, y_size)) for (img, point) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == goldlabel_types.number_of_lines:
        data = [downscale(img, points, 4, 2) for (img, points) in data]  # TODO should be optional
        #h = int(h/2)
        #w = int(w/4)
        data = [fitsize(img, gl, w, h) for (img, gl) in data]
        maxlinecount = max([gl for (img, gl) in data])
        if goldlabel_encoding == goldlabel_encodings.dense:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), lc/maxlinecount) for (img, lc) in data]
        elif goldlabel_encoding == goldlabel_encodings.onehot:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), num2sparse(lc, maxlinecount)) for (img, lc) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    else:
        raise "invalid goldlabel_type: "+str(goldlabel_type)


def concat_data(data_sublist, goldlabel_type, axis=0, pad=None):
    """
    :param data_sublist:
        goldlabels have to be list of length 1, that means only words and lines, no paragraphs are valid input.
    :param goldlabel_type:
    :param axis:
        axis == 0: images of data_sublist are appendend below each other.
        axis == 1: images of data_sublist are appendend right of other.
    :param pad:
        pad[i] = amount of pad inserted befor word/line i
    :return:
        an element of the new data list
    """
    # axis=0 -> bilder untereinander -> img.shape[1] muss gleich sein
    # img.shape = (height, width) of img
    # reshape all images to same size
    if pad is None:
        pad = [0]*len(data_sublist)
    assert len(pad) == len(data_sublist)
    img_list = [d[0] for d in data_sublist]
    goldlabel_list = [d[1] for d in data_sublist]  # [goldlabel:[((int, int), (int, int), int)]], with len(goldlabel)=1
    #print("goldlabel_list = ", goldlabel_list)
    goldlabel = None
    if axis == 0:
        w = max([img.shape[1] for img in img_list])
        # np.pad(np-array, ((oben, unten), (links, rechts)
        img_list = [np.pad(img_list[i], ((pad[i], 0), (0, w-img_list[i].shape[1])), mode='constant', constant_values=255) for i in range(len(img_list))]  # white padding added at right margin
        if goldlabel_type == goldlabel_types.text:
            goldlabel = '<br>'.join(goldlabel_list)
        elif goldlabel_type == goldlabel_types.linepositions:
            goldlabel = [((0, 0), (0, 0), 0)]*len(goldlabel_list)
            offset = 0
            for i_gl in range(len(goldlabel_list)):  # gl is list of points
                #print("goldlabe_list[", i_gl, "] = ", str(goldlabel_list[i_gl]))
                pre_gl = goldlabel_list[i_gl][0]  # (startpoint, endpoint, height)
                goldlabel[i_gl] = ((pre_gl[0][0], pre_gl[0][1]+offset), (pre_gl[1][0], pre_gl[1][1]+offset), pre_gl[2])
                offset += pad[i_gl]+pre_gl[2]
        else:
            print("Dataloader.concat_data: goldlabel_type ", goldlabel_type, " is not valid")
    elif axis == 1:
        h = max([img.shape[0] for img in img_list])
        img_list = [np.pad(img_list[i], ((0, h-img_list[i].shape[0]), (pad[i], 0)), mode='constant', constant_values=255) for i in range(len(img_list))]  # white padding added at bottom
        if goldlabel_type == goldlabel_types.text:
            goldlabel = ' '.join(goldlabel_list)
        elif goldlabel_type == goldlabel_types.linepositions:
            # line start at start point of first word, ends at endpoint of last word + sum(width every other word), and has the hight of maxium height of each word.
            startpoint = goldlabel_list[0][0][0]  # (float, float)  # startpoint of line is startpoint of first word.
            endpoint = goldlabel_list[-1][0][1]  # (float, float)  # endpoint of line is endpoint of last word
            widths = [abs(point[0][1][0]-point[0][0][0]) for point in goldlabel_list]
            endpoint = (startpoint[0]+sum(widths)+sum(pad), endpoint[1])
            hight = max([gl[0][2] for gl in goldlabel_list])  # float
            goldlabel = [(startpoint, endpoint, hight)]
        else:
            print("Dataloader.concat_data: goldlabel_type ", goldlabel_type, " is not valid")
    else:
        print("axis should be 0 or 1.")
        return None
    return (np.concatenate(img_list, axis=axis), goldlabel)


def load_iam(dir=iam_dir, goldlabel_type=goldlabel_types.linepositions):
    """
    :param dir:
    directory of the iam dataset
    only uses the word pictures of iam, lines, sentences and paragraphs are unused.
    :param goldlabel_type:
    element of goldlabel_types that indicates what the goldlabel of a text should be
    :return:
    data: [(relative image path, goldlabel: [(start_of_line: (int, int), end_of_line: (int, int), height: int)])]
    """
    # format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
    #
    #     a01-000u-00-00  -> word id for line 00 in form a01-000u
    #     ok              -> result of word segmentation
    #                            ok: word was correctly
    #                            er: segmentation of word can be bad
    #
    #     154             -> graylevel to binarize the line containing this word
    #     408 768 27 51   -> bounding box around this word in x,y,w,h format, with (x, y) = ? and (w, h) img.shape
    #     AT              -> the grammatical tag for this word, see the
    #                        file tagset.txt for an explanation
    #     A               -> the transcription for this word
    bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # known broken images in IAM dataset, thx SimpleHTR
    word_img_gt = []
    with open(dir+"iam/gt/words.txt") as f:
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')  # see comment at start of dir+"/gt/words.txt" for information
            assert len(line_split) >= 9
            if len(line_split) != 9:
                #print("line_split = ", line_split)
                continue
            if line_split[0] in bad_samples_reference:
                continue
            # line_split[0] ="a01-000u-00-00"
            # img_filename = "img\a01\a01-000u\a01-000u-00-00"
            img_filename_split = line_split[0].split("-")
            img_filename = "iam/img/"+img_filename_split[0]+"/"+img_filename_split[0]+"-"+img_filename_split[1]+"/"+line_split[0]+".png"
            if goldlabel_type == goldlabel_types.text:
                goldlabel = ' '.join(line_split[8:])
            elif goldlabel_type == goldlabel_types.linepositions:
                width = int(line_split[5])
                hight = int(line_split[6])
                goldlabel = [((0, int(0.5*hight)), (width, int(0.5*hight)), hight)]
            elif goldlabel_type == goldlabel_types.number_of_lines:
                goldlabel = 1
            else:
                goldlabel = "TODO: invalid goldlabel_type in Dataloader.load_iam: "+str(goldlabel_type)
            word_img_gt.append((img_filename, goldlabel))
    return word_img_gt

def store(img_gl_data, dir=dataset_dir):
    """
    :param img_gl_data:
    [(img: nparray, goldlabel) data that gets written to dir
    :param dir:
    direktory in that the data gets written
    :return:
    writes image_path - str(goldlabel) in dir_gl.txt
    """
    with open(dir+"/dir-gl.txt", 'w') as f:
        i = 0
        for (img, gl) in img_gl_data:
            tpath = dir+"/"+str(i)+".png"
            f.write(tpath+"-"+str(gl)+"\n")
            cv2.imwrite(tpath, img)
            i += 1
    return None



def getTrainingData(goldlabel_encoding=goldlabel_encodings.onehot):
    """
    :param goldlabel_encoding:
    from Dataset.goldlabel_encodings
    :return:
    (x_train, y_train), (x_val, y_val), (x_test), (y_test), so that tf.model.fit(x_train, y_train, validation_data=(x_val, y_val)) works.
    """
    data = getData(dir=data_dir, dataset_loader=dataset_names.iam, img_type=img_types.paragraph, goldlabel_type=goldlabel_types.linepositions, goldlabel_encoding=goldlabel_encoding, maxcount=100, x_size=(512, 1024))
    train_val_split = int(0.8*len(data))  # 80% training, 10% validation, 10% test
    val_test_split = int(0.9*len(data))
    print("split: ", train_val_split, " : ", val_test_split, " : ", len(data))
    data_train = data[:train_val_split]
    data_val = data[train_val_split:val_test_split]
    data_test = data[val_test_split:]

    x_train = np.array([d[0] for d in data_train], dtype=float)
    y_train = np.array([d[1] for d in data_train], dtype=float)
    x_val = np.array([d[0] for d in data_val], dtype=float)
    y_val = np.array([d[1] for d in data_val], dtype=float)
    x_test = np.array([d[0] for d in data_test], dtype=float)
    y_test = np.array([d[1] for d in data_test], dtype=float)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def getData(dir, dataset_loader=dataset_names.iam, img_type=img_types.paragraph, goldlabel_type=goldlabel_types.text, goldlabel_encoding=goldlabel_encodings.onehot, x_size = (100, 100), maxcount=-1, offset=0):
    """
    :param dataset_name:
        name of the dataset to use, currently only \"iam\" supported
    :param img_type:
        type of images to train on: word, line, paragraph
    :param maxcount:
        maximum for len(trainingsdata)
    :return:
        trainingsdata: list of (grayscale imag, goldlabel text)-tupel
    """
    #TODO check if dir exists and contains the correct data.
    if img_type not in [vars(img_types)[x] for x in vars(img_types).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_loader, ", ", img_type, ", ", maxcount, "): invalid input, img_type should be img_types.*")
        return None
    if dataset_loader not in [vars(dataset_names)[x] for x in vars(dataset_names).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_loader, ", ", img_type, ", ", maxcount, "): invalid input, dataset should be dataset_names.iam")
        return None
    if goldlabel_type not in [vars(goldlabel_types)[x] for x in vars(goldlabel_types).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_loader, ", ", img_type, ", ", maxcount, "): invalid input, goldlabel_type should be ", goldlabel_types)
        return None
    words_per_line = [2, 3]
    lines_per_paragrph = [2, 3, 4]

    if dataset_loader == dataset_names.iam:
        data = load_iam(dir, goldlabel_type)  # [(relative path of img file, goldlabel text of that file)]
    else:
        raise "invalid dataset_loader: "+str(dataset_loader)
    print("path_gl: ", getType(data))
    #print("data_imgpath_goldlabel = ", data[:5])
    # if goldlabel_type = text: type(data) = [(img: np.array(?, ?), text: string)]
    # if goldlabel_type = linepositions: type(data) = [(img: np.array(?, ?), [point: (int, int)])]

    if 0 < maxcount < len(data) and offset+maxcount < len(data):
        if img_type == img_types.word:
            maxcount = maxcount
        elif img_type == img_types.line:
            maxcount = int(maxcount*max(words_per_line))
        elif img_type == img_types.paragraph:
            maxcount = int(maxcount*max(words_per_line)*max(lines_per_paragrph))
        else:
            print("unexpected img_type: ", img_type)
            return None
        data = data[offset:offset+maxcount]
    print("path_gl_short: ", getType(data))
    data = [(load_img(dir+"/"+path), gl) for (path, gl) in data]
    print("imgword_gl: ", getType(data))
    if img_type == img_types.word:
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size)
        print("imgwordenc_gl: ", getType(data))
        return data

    tmp = [data[i:i+random_choice(words_per_line)] for i in range(0, len(data), max(words_per_line))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    word_distance = [10, 20]
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=1, pad=[random_choice(word_distance) for unused in range(len(t))]) for t in tmp]
    #print("data_lines[0]: ", data[0])
    #print("data_imgline_goldlabel = ", data[:5])
    print("imgline_gl: ", getType(data))
    if img_type == img_types.line:  # line
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=1)
        print("imglineenc_gl: ", getType(data))
        return data

    tmp = [data[i:i+random_choice(lines_per_paragrph)] for i in range(0, len(data), max(lines_per_paragrph))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    line_distance = [5, 10]
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=0, pad=[random_choice(line_distance) for unused in range(len(t))]) for t in tmp]
    print("imgpara_gl: ", getType(data))
    #print("data_parag[0]: ", data[0])
    #print("data_imgpara_goldlabel = ", data[:5])
    if img_type == img_types.paragraph:  # paragraph
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=max(lines_per_paragrph))
        print("imgparaenc_gl: ", getType(data))
        return data
    return "Dataloader.getData: This return statement is impossible to reach."

class Dataset_test:
    def __init__(self, difficulty=0):
        self.diff = difficulty

    def get_batch(self, n):
        """
        :param n:
        the number of elements in the batch returned
        :return:
        a data batch ([img], [gl]), that can directly be used for tf.model.fit(x=[img], y=[gl], ...)
        in this dataset, the images have size 32x32 and show one or multiply circles.
        the goldlabel for each image is the position of the circles.
        """
        size = 32
        r = []
        poses = [(int(i/size), i%size) for i in range(size**2)]
        random.shuffle(poses)
        for i in range(n):
            img = np.full((size, size), 255, dtype='uint8')
            if self.diff == 0:
                # ein kreis
                pos = [poses[i]]
                for p in pos:
                    cv2.circle(img, p, 5, 0, thickness=5)
            elif self.diff == 1:
                # zwei kreis, einer links, anderer rechts
                pos = [random.choice(poses) for unused in range(2)]
                hs = int(size/2)
                pos = [(pos[0][0]%hs, pos[0][1]), (pos[1][0]%hs+hs, pos[1][0])]
                for p in pos:
                    cv2.circle(img, p, 5, 0, thickness=5)
            elif self.diff == 2:
                # zwei kreis
                pos = [random.choice(poses) for unused in range(2)]
                for p in pos:
                    cv2.circle(img, p, 5, 0, thickness=5)
            elif self.diff == 3:
                # random(3) kreis
                tmp = random.choice(list(range(4)))
                pos = [random.choice(poses) for unused in range(tmp)]
                for p in pos:
                    cv2.circle(img, p, 5, 0, thickness=5)
                pos += [(0, 0)]*(3-tmp)
            else:
                print("unsuported difficulty: ", self.diff)
                return None
            pos.sort(key=lambda x: x[0])
            r.append((img, points2dense(pos)))
        return np.array([d[0] for d in r]), np.array([d[1] for d in r])  # img, goldlabel

    def show(self, batch):
        """
        :param batch:
        :type ([img], [goldlabels])
        with img is type as in opencCV
        and goldlabel is [float]
        e.g. the return value of self.get_batch:
        x, y = dataset.get_batch(20)
        dataset.show((x, y))
        :return:
        None, but shows (cv2.imshow) an represantation of the data
        """
        assert len(batch[0]) == len(batch[1])
        batch = [(np.array(batch[0][i], dtype="uint8"), dense2points(batch[1][i])) for i in range(len(batch[0]))]
        for (img, gl) in batch:
            print("test_dataset.show: img: "+str(img)+" -> "+str(gl))
            for point in gl:
                cv2.circle(img, point, 5, 125, 3)
            cv2.imshow(str(gl), img)
            cv2.waitKey(0)

class Dataset:
    data_dir = None
    gl_type = 0
    gl_encoding = 0
    pos = 0
    dataset_size = -1
    imgsize = (256, 256)

    def __init__(self, datadir, gl_type, gl_encoding):
        self.data_dir = datadir
        self.gl_type = gl_type
        self.gl_encoding = gl_encoding
        self.pos = 0
        self.dataset_size = len(load_iam(datadir, gl_type))

    def get_batch(self, size):
        #data = get_testdata()
        #return np.array([d[0] for d in data]), np.array([d[1] for d in data])
        #selekt witch part of dset to use
        assert size < self.dataset_size
        if self.pos+size >= self.dataset_size:
            self.pos = self.pos % (self.dataset_size-size)
        data = getData(dir=data_dir, dataset_loader=dataset_names.iam, img_type=img_types.paragraph, goldlabel_type=self.gl_type, goldlabel_encoding=self.gl_encoding, maxcount=size, x_size=self.imgsize)
        self.pos = self.pos+size
        x_train = np.array([d[0] for d in data], dtype=float)
        y_train = np.array([d[1] for d in data], dtype=float)
        return x_train, y_train

    def show(self, batch):
        h, w = batch[0][0].shape
        print("main.show_points_data: w, h = ", h, ", ", w)
        if max([max(p) for (img, p) in batch]) > 1.1 or min([min(p) for (img, p) in batch]) < -0.1:
            print("main.show_points_data: point outside of bounds. points = ", [points for (img, points) in batch])
        for (img, points) in batch:
            img = np.array(img, dtype="uint8")
            print("show_points_data.img: ", batch.getType(img))
            print("show_points_data.pts: ", batch.getType(points))
            print("show_points_data.pts: ", points)
            if self.gl_encoding == goldlabel_encodings.dense:
                points = dense2linepoints(points, max_x=w, max_y=h)
            elif self.gl_encoding == goldlabel_encodings.onehot:
                points = sparse2linepoints(points, max_x=w, max_y=h)
            else:
                print("TODO unsupported gl_encoding in dataset.show: ", self.gl_encoding)
                points = [((0, 0), (0, 0), 0)]
            for point in points:
                #print("point = ", point)
                point = ((max(1, int(point[0][0])), max(1, int(point[0][1]))), (max(1, int(point[1][0])), max(1, int(point[1][1]))), max(1, int(point[2])))
                cv2.circle(img, point[0], int(point[2]/2), 125, 2)
                cv2.rectangle(img, (point[0][0]-5, point[0][1]-5), (point[0][0]+5, point[0][1]+5), 125, 2)
                cv2.circle(img, point[1], int(point[2]/2), 125, 2)
                cv2.line(img, point[0], point[1], 125, thickness=1)
            cv2.imshow(str(points), img)
        cv2.waitKey(0)

# TODO abgeschnittene Zeilen, leere Seiten einfügen

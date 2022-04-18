import cv2
import numpy as np
import tensorflow as tf
import random
# import tensorflow_datasets as tf_ds
# TODO https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# TODO tf.keras.utils.image_dataset_from_directory nutzen
# 125 GB RAM auf server (gesamt)

line_point = ((int, int), (int, int), int)  # (startpoint of line, endpoint of line, height)

data_dir = "../SimpleHTR/data/trainingDataset/"  # The dirctory that is mapped to not be in the docker
#data_dir = "C:/Users/Idefix/PycharmProjects/SimpleHTR/data/"
iam_dir = data_dir + "iam/"  # the unchanged iam dataset
dataset_dir = data_dir + "generated/"  # directoy for storing generated data
models_dir = data_dir + "models/"  # directoy for storing trained models


class GoldlabelTypes:
    text = 0  # str the text that is written in the image, e.g. "A move to Stop mr. "
    linepositions = 1  # (start_of_line:(x:int, y:int), end_of_line:(x2:int, y2:int), height_of_line:int) all information needed to manually segment the line out of the image
    number_of_lines = 2  # int
    #add new type: linemask: [image where exactly the pixels of line i are 1, everything else 0 for i in range(max(lines_per_paragrph))]
    # attention layer maybe supports masking
    # masking may only support RNN and only one bool per timestamp https://www.tensorflow.org/guide/keras/custom_layers_and_models
    # image segmentation (NN that predicts label for each pixel) https://www.tensorflow.org/tutorials/images/segmentation
    lineimg = 4  # list of all textlines, each with the same size (32x256), only supports dense encoding


class GoldlabelEncodings:
    onehot = 10
    dense = 11


class ImgTypes:
    word = 20
    line = 21
    paragraph = 22


class DatasetNames:
    iam = 30


@tf.function
def extractline(img, linepoint: [float], max_x: int, max_y: int):
    """
    :param img:
    :param linepoint:
    output from Dataloader.linepoints2dense
    :return:
    the normalised part of the image where the text line described by linepoint should be
    """
    assert len(linepoint) == 5
    img = np.array(img, dtype="uint8")

    #print("Datloader.extractline: linepoint = ", linepoint)
    linepoint = dense2linepoints(linepoint, max_x=max_x, max_y=max_y)
    #print("Datloader.extractline: linepoint = ", linepoint)
    #rotate image so that line is horizontal
    ((x1, y1), (x2, y2), h_lp) = linepoint[0]
    # y1 == y2 => ist bereits gerade
    if abs(x2-x1)+abs(y2-y1) < 5:
        # <=> is empty  ((0, 0), (0, 0), 0) linepoint
        print("Dataloader.extractline: empty linepoint -> return black empty image")
        return np.zeros((32, 128))
    if abs(y2-y1) > 1:  # line is not already horizontal
        alpha = np.arcsin((y2-y1)/(np.sqrt((x2-x1)**2+(y2-y1)**2)))
        #print("Dataloader.extractline:  alpha = ", alpha)
        #print("Dataloader.extractline: y1 == y2: ", y1, " == ", y2)
        img, linepoint = rotate_img(img, linepoint, GoldlabelTypes.linepositions, angle=int(alpha*(180/np.pi)))
        ((x1, y1), (x2, y2), h_lp) = linepoint[0]
        #print("Dataloader.extractline: y1 == y2: ", y1, " == ", y2)
    # cut line
    #cv2.imshow("rotated", img)
    #cv2.waitKey(0)
    print("Dataloader.extractline: linepoint = ", linepoint)
    (h_img, w) = img.shape
    print("Dataloader.extractline: img.shape = ", (h_img, w))
    y = int(0.5*y1+0.5*y2)  # y1 and y2 should already be almost the same
    left_bound = max(0, int(x1-h_lp))
    right_bound = min(w, int(x2+h_lp))
    upper_bound = max(0, int(y-h_lp))
    lower_bound = min(h_img, int(y+h_lp))
    #print("Dataloader.extractline: bounds = ", ((left_bound, right_bound), (upper_bound, lower_bound)))
    img = np.array(img[left_bound:right_bound][upper_bound:lower_bound], dtype="uint8")
    # scale image to hight of 32
    print("Dataloader.extractline: img.shape = ", img.shape)
    return cv2.resize(img, (128, 32), interpolation=cv2.INTER_AREA)


# <debug functions>


def getType(x):
    deliminating_chars = {"list": ('[', ';', ']'), "tupel": ('<', ';', '>'), "dict": ('{', ';', '}')}
    name = type(x).__name__
    if name == 'list':
        return '['+str(len(x))+":"+getType(x[0])+']'  # assumes all element of the list have the same type
    if name == 'tuple':
        r = '<'+str(len(x))+":"
        for i in x:
            r += getType(i)+'; '
        return r[:-2] + '>'
    if name == 'dict':
        r = "{"+str(len(x.keys()))+":"
        for key in x.keys():
            r += getType(key)+": "+getType(x[key])+"; "  # would be more in line with other types
            #r += str(key)+": "+getType(x[key])+"; "  # contains more information
        r = r[:-1]+" }"
        return r
    if name == 'ndarray':
        return 'ndarray('+str(x.shape)+': '+(str(x.dtype) if len(x) > 0 else "Nix")+')'
    if name == 'BatchDataset':
        return str(name)+" : "+str(len(x))
    if name in ["KerasTensor", "Tensor", "EagerTensor"]:
        return str(name)+"("+str(x.shape)+":"+str(x.dtype)+")"
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


#</debug functions>

def apply_rotmat(rotmat, point):
    [[a00, a01, b0], [a10, a11, b1]] = rotmat
    (x, y) = point
    return a00*x+a01*y+b0, a10*x+a11*y+b1


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


def num2sparse(num, max):
    r = [0]*max
    r[num] = 1
    return r
def sparse2num(sparse):
    r = 0
    for i in range(len(sparse)):
        if sparse[i] > sparse[r]:
            r = i
    return r


def points2dense(points: [(int, int)], max_x=32, max_y=32) -> [float]:
    return [points[int(i/2)][i%2]/max_x for i in range(len(points)*2)]
def dense2points(points: [float], max_x=32, max_y=32) -> [(int, int)]:
    return [(int(points[2*i]*max_x), int(points[2*i+1]*max_x)) for i in range(int(len(points)/2))]


alphabet = np.array([""]+list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMOPQRSTUVWXYZ1234567890.,:;!#"))
def txt2sparse(txt, y_size):
    #print("Dataloader.txt2sparse: txt ", len(txt), ", ysize ", y_size)
    assert len(txt) <= y_size
    #txt = txt.lower()
    a = np.array([(np.argwhere(alphabet == c)[0][0] if c in alphabet else 0) for c in txt])
    #b = np.zeros(y_size*len(alphabet))
    b = np.zeros((y_size, len(alphabet)))
    for i in range(len(a)):
        #b[i*len(alphabet)+a[i]] = 1
        b[i][a[i]] = 1
    return np.array(b)
def sparse2txt(sparse, y_size):
    dense = [str(alphabet[np.argmax(sparse_char)]) for sparse_char in sparse]
    return "".join(dense)


def txt2dense(txt, y_size):
    return np.array([(np.argwhere(alphabet == c)[0][0] if c in alphabet else 0) for c in (txt.lower()+" "*(y_size-len(txt)))])
def dense2txt(txt, y_size):
    return "".join([str(alphabet[t]) for t in txt])


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
    return [((int(points[i]*max_x), int(points[i+1]*max_y)), (int(points[i+2]*max_x), int(points[i+3]*max_y)), max(1, int(points[i+4]*max_y))) for i in range(0, len(points), 5)]


def fitsize_linepoint(img, goldlabel: [line_point], upperleftcorner: (int, int), sampesize: (int, int)):
    """
    cuts an area out of the image.
    currently only used in fitsize
    :param img:
    an oopencv-img, that means numpy-array
    :param upperleftcorner:
    the upper left corner of the return image in the input image
    :param sampesize:
    the size of the return image
    :return:
    an image that shows an area out of th input image and the edited goldlabel
    """
    #print("Dataloader.sample_linepoint")
    ulx, uly = upperleftcorner  # up-left corner
    drx = ulx+sampesize[0]  # down-right corner
    dry = uly+sampesize[1]
    glr = []
    for ((x1, y1), (x2, y2), h) in goldlabel:
        # if completly out of range: remove
        if uly > min(y1, y2)+1 and max(y1, y2) > dry+1:
            continue
        # if cutted: move gl to be in bounds
        x1 = min(max(x1, ulx), drx)
        x2 = min(max(x2, ulx), drx)
        y1 = min(max(y1, uly), dry)
        y2 = min(max(y2, uly), dry)
        h = min(h, abs(y1-uly), abs(y1-dry), abs(y2-uly), abs(y2-dry))
        glr.append(((x1-ulx, y1-uly), (x2-ulx, y2-uly), h))
    if len(glr) == 0:
        glr = [((0, 0), (0, 0), 0)]
    return np.array(img[uly:dry, ulx:drx], dtype="uint8"), glr

def downscale(img, goldlabel, x: int, y: int, gl_type=GoldlabelTypes):
    """
    reduces the resolution of the image
    :param img:
    an opencv-img, that means numpy-array
    :param goldlabel:
    the goldlabel of image, unencoded
    :param gl_type:
    the type of the goldlabel, as defined in Dataloader.GoldlabelType
    :param x:
    the factor by witch the width of the image is scaled
    :param y:
    the factor by witch the hight of the image is scaled
    :return:
    an image of size (input_size[0]/x, input_size[1]/y)
    """
    img = cv2.resize(img, dsize=None, fx=1/x, fy=1/y)
    # postsize[0] = 1/y*praesize[0]
    # postsize[1] = 1/x*praesize[1]
    if gl_type == GoldlabelTypes.linepositions:
        goldlabel = [((int(x1/x), int(y1/y)), (int(x2/x), int(y2/y)), int(h/y)) for ((x1, y1), (x2, y2), h) in goldlabel]
    elif gl_type == GoldlabelTypes.lineimg:
        #print("Dataloader.downscale: goldlabel = ", getType(goldlabel))
        goldlabel = np.array(goldlabel)
        if len(goldlabel.shape) == 2:  # goldlabel is image of single line
            goldlabel = cv2.resize(goldlabel, (256, 32))
        else:  # goldlabel ist list of images
            goldlabel = [cv2.resize(np.array(gl), (256, 32)) for gl in goldlabel]
    return np.array(img, dtype="uint8"), goldlabel


def fitsize(img, gl, w, h, gl_type):
    """
    :param img:
    :param gl:
    :param w:
    the new width of the image
    :param h:
    the new height of the image
    :param gl_type:
    element of GoldlabelTypes
    :return:
    the image and goldlabel, so that they still are consistent and the image has exactly the shape (h, w)
    """
    #print("Dataloader.fitsiz: (h, w) = ", (h, w))
    if gl_type == GoldlabelTypes.linepositions:
        if img.shape[0] >= h:
            img, gl = fitsize_linepoint(img, gl, (0, 0), (img.shape[0], h-1))
        if img.shape[1] >= w:
            img, gl = fitsize_linepoint(img, gl, (0, 0), (w-1, img.shape[1]))
    elif gl_type == GoldlabelTypes.text:
        img = np.array(img[0:h, 0:w], dtype="uint8")  # TODO text that is cutted out of the image should be cut out of the goldlabel
    elif gl_type == GoldlabelTypes.lineimg:
        img = np.array(img[0:h, 0:w])
        gl = [np.array(glimg[0:32, 0:256]) for glimg in gl]  # TODO size of lineimg should not be hardcoded
        #print("Dataloader.fitsize: gl shape: ", [glimg.shape for glimg in gl])
        gl = [np.pad(glimg, ((0, 32-glimg.shape[0]), (0, 256-glimg.shape[1])), mode='constant', constant_values=255) for glimg in gl]
    else:
        print("Invalid goldlabelencoding in Dataloader.fitsize: "+str(gl_type))
        raise "Invalid goldlabelencoding in Dataloader.fitsize: "+str(gl_type)
    img = np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255)
    #print("fitsize: (h, w) = ", (h, w), "shape = ", img.shape)
    assert img.shape == (h, w)
    return img, gl


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
    that means they either get padded or downscaled and clipped.
    :param y_size:
    size all goldlabel_encodings in data will have
    :return:
    [(image, goldlabel)] like input, but the goldlabel is encoded and image and goldlabel are padded to size
    """

    if size is None:
        h = int(max([img.shape[0] for (img, gl) in data]))
        w = int(max([img.shape[1] for (img, gl) in data]))
        #print("Dataloader.encode_and_pad: shape in w, h = ", h, ", ", w)
    else:
        (h, w) = size
        hn = int(max([img.shape[0] for (img, gl) in data]))
        wn = int(max([img.shape[1] for (img, gl) in data]))
        #print("Dataloader.encode_and_pad: (hn, wn) = ", (hn, wn), " == ", (h, w), " = (h, w) = size")
        data = [downscale(img, gl, y=img.shape[0]/h, x=img.shape[1]/w, gl_type=goldlabel_type) for (img, gl) in data]
        # postsize[0] = 1/y*praesize[0]
        # postsize[1] = 1/x*praesize[1]
        hn = int(max([img.shape[0] for (img, gl) in data]))
        wn = int(max([img.shape[1] for (img, gl) in data]))
        #print("Dataloader.encode_and_pad: (hn, wn) = ", (hn, wn), " == ", (h, w), " = (h, w) = size")
    #print("Dataloader.encode_and_pad: used w, h = ", h, ", ", w)
    data = [fitsize(img, gl, w, h, gl_type=goldlabel_type) for (img, gl) in data]
    if goldlabel_type == GoldlabelTypes.text:

        if goldlabel_encoding == GoldlabelEncodings.dense:
            print("dense text encoding is currently unsupported, return text in UTF-8 as goldlabel instead")
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), txt2dense(txt, y_size)) for (img, txt) in data]  # padding should already be met by fitsize
        elif goldlabel_encoding == GoldlabelEncodings.onehot:  # text in one-hot
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), txt2sparse(txt, y_size)) for (img, txt) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == GoldlabelTypes.linepositions:
        if goldlabel_encoding == GoldlabelEncodings.dense:
            data = [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), linepoint2dense(point, max_x=w, max_y=h, y_size=y_size)) for (img, point) in data]
            return data
        elif goldlabel_encoding == GoldlabelEncodings.onehot:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), linepoint2sparse(point, w, h, y_size)) for (img, point) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == GoldlabelTypes.number_of_lines:
        maxlinecount = max([gl for (img, gl) in data])
        if goldlabel_encoding == GoldlabelEncodings.dense:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), lc/maxlinecount) for (img, lc) in data]
        elif goldlabel_encoding == GoldlabelEncodings.onehot:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), num2sparse(lc, maxlinecount)) for (img, lc) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == GoldlabelTypes.lineimg:
        # does not support onehot encoding, uses dense in any case
        # rescale img from [0, 255] to [0, 1]:
        return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255),  lineimgs) for (img, lineimgs) in data]
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
        if goldlabel_type == GoldlabelTypes.text:
            goldlabel = ''.join(goldlabel_list)
        elif goldlabel_type == GoldlabelTypes.linepositions:
            goldlabel = [((0, 0), (0, 0), 0)]*len(goldlabel_list)
            offset = 0
            for i_gl in range(len(goldlabel_list)):  # gl is list of points
                #print("goldlabe_list[", i_gl, "] = ", str(goldlabel_list[i_gl]))
                pre_gl = goldlabel_list[i_gl][0]  # (startpoint, endpoint, height)
                goldlabel[i_gl] = ((pre_gl[0][0], pre_gl[0][1]+offset), (pre_gl[1][0], pre_gl[1][1]+offset), pre_gl[2])
                #offset += pad[i_gl]+pre_gl[2]
                offset += img_list[i_gl].shape[0]
        elif goldlabel_type == GoldlabelTypes.lineimg:
            # axis == 0 -> bilder untereinander -> img_list is list of lineimg
            goldlabel = goldlabel_list
        else:
            print("Dataloader.concat_data: goldlabel_type ", goldlabel_type, " is not valid")
    elif axis == 1:
        h = max([img.shape[0] for img in img_list])
        img_list = [np.pad(img_list[i], ((int(np.floor_divide((h-img_list[i].shape[0]),2)), int(np.ceil((h-img_list[i].shape[0])/2))), (pad[i], 0)), mode='constant', constant_values=255) for i in range(len(img_list))]  # white padding added at bottom
        if goldlabel_type == GoldlabelTypes.text:
            goldlabel = ' '.join(goldlabel_list)+"#"  # # is end-of-line-symbol
        elif goldlabel_type == GoldlabelTypes.linepositions:
            # line start at start point of first word, ends at endpoint of last word + sum(width every other word), and has the hight of maxium height of each word.
            # goldlabel = [((x1, y1), (x2, y2), h)]
            # goldlabel_list = [goldlabel]
            #print("goldlabel_list = ", getType(goldlabel_list))  # goldlabel_list =  [3:[1:<3:<2:int; int>; <2:int; int>; int>]]
            hight = max([gl[0][2] for gl in goldlabel_list])  # float
            startpoint = (goldlabel_list[0][0][0][0], goldlabel_list[0][0][0][1]+np.floor_divide(hight-goldlabel_list[0][0][2], 2))
            endpoint = (goldlabel_list[-1][0][1][0], goldlabel_list[-1][0][1][1]+np.floor_divide(hight-goldlabel_list[-1][0][2], 2))
            #endpoint = goldlabel_list[-1][0][1]
            widths = [abs(point[0][1][0]-point[0][0][0]) for point in goldlabel_list]
            endpoint = (startpoint[0]+sum(widths)+sum(pad), endpoint[1])
            goldlabel = [(startpoint, endpoint, hight)]
        elif goldlabel_type == GoldlabelTypes.lineimg:
            #print("Dataloader.concat_data: gl_list shapes = ", [gl.shape for gl in goldlabel_list])
            mh = max([gl.shape[0] for gl in goldlabel_list])
            #print("Dataloader.concat_data: gl_list shapes = ", [gl.shape for gl in goldlabel_list])
            goldlabel_list = [np.pad(gl, ((0, mh-gl.shape[0]), (0, 0)), mode='constant', constant_values=255) for gl in goldlabel_list]
            #print("Dataloader.concat_data: gl_list = ", getType(goldlabel_list))
            #print("Dataloader.concat_data: gl_list = ", [gl.shape for gl in goldlabel_list])
            goldlabel = np.concatenate(goldlabel_list, axis=1)
        else:
            print("Dataloader.concat_data: goldlabel_type ", goldlabel_type, " is not valid")
    else:
        print("axis should be 0 or 1.")
        return None
    return np.concatenate(img_list, axis=axis), goldlabel


def load_iam(datadir=iam_dir, goldlabel_type=GoldlabelTypes.linepositions):
    """
    :param datadir:
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
    with open(datadir+"iam/gt/words.txt") as f:
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split(' ')  # see comment at start of dir+"/gt/words.txt" for information
            assert len(line_split) >= 9
            if len(line_split) != 9:
                # print("line_split = ", line_split)
                continue
            if line_split[0] in bad_samples_reference:
                continue
            # line_split[0] ="a01-000u-00-00"
            # img_filename = "img\a01\a01-000u\a01-000u-00-00"
            img_filename_split = line_split[0].split("-")
            img_filename = "iam/img/"+img_filename_split[0]+"/"+img_filename_split[0]+"-"+img_filename_split[1]+"/"+line_split[0]+".png"
            if goldlabel_type == GoldlabelTypes.text:
                goldlabel = ' '.join(line_split[8:])
            elif goldlabel_type == GoldlabelTypes.linepositions:
                width = int(line_split[5])
                hight = int(line_split[6])
                goldlabel = [((0, int(0.5*hight)), (width, int(0.5*hight)), hight)]
            elif goldlabel_type == GoldlabelTypes.number_of_lines:
                goldlabel = 1
            elif goldlabel_type == GoldlabelTypes.lineimg:
                goldlabel = None
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


def rotate_img(image, gl, gl_type, angle):
    """
    :param image:
    an opencv-image
    :param gl:
    goldlabel in unencoded form
    :param gl_type:
    type of gl. if linepoint, the points will be transformed the same way as the image.
    otherwise gl will stay the same
    :param angle:
    the angle in degrees by witch the image gets rotated.
    Positive values mean counter-clockwise rotation
    :return:
    the image, but rotated and scaled down so the complete image fits in the same shape
    """
    #copied from https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    sin = abs(np.sin(angle*(2*np.pi/360)))
    cos = abs(np.cos(angle*(2*np.pi/360)))
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    #print("Dataloader.rotate_img: (w, h) = ", (w, h))
    #print("Dataloader.rotate_img: (sin, cos) = ", (sin, cos))
    #print("Dataloader.rotate_img: (nW, nH) = ", (nW, nH))
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1)
    # compute the new bounding dimensions of the image

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    #update the goldlabel
    if gl_type == GoldlabelTypes.linepositions:
        #print("Dataloader.rotate_img: gl = ", gl)
        gl = [(apply_rotmat(M, (x1, y1)), apply_rotmat(M, (x2, y2)), h) for ((x1, y1), (x2, y2), h) in gl]
        #gl = [(np.multiply(apply_rotmat(M, (x1, y1)), scale), np.multiply(apply_rotmat(M, (x2, y2)), scale), h*scale) for ((x1, y1), (x2, y2), h) in gl]
        #print("Dataloader.rotate_img: gl = ", gl)
        gl = [((int(x1), int(y1)), (int(x2), int(y2)), int(h)) for ((x1, y1), (x2, y2), h) in gl]
    elif gl_type == GoldlabelTypes.text:
        pass
    elif gl_type == GoldlabelTypes.number_of_lines:
        pass
    elif gl_type == GoldlabelTypes.lineimg:
        pass
    else:
        print("Dataloader.rotate_img: TODO unsupported goldlabel_type in Dataloader.rotate_img: ", gl_type)
        pass
    # rescale image to have original size
    scale = min(nW/w, nH/h)
    image, gl = downscale(image, gl, x=scale, y=scale, gl_type=gl_type)
    return image, gl


def getData(dir, dataset_loader=DatasetNames.iam, img_type=ImgTypes.paragraph, goldlabel_type=GoldlabelTypes.text, goldlabel_encoding=GoldlabelEncodings.onehot, x_size = (100, 100), maxcount=-1, offset=0, line_para_winkel=(2, 5)):
    """
    :param img_type:
        type of images to train on: word, line, paragraph (in Dataloader.ImageType)
    :param maxcount:
        maximum for len(trainingsdata)
    :param line_para_winkel:
        (maximum angel by that lines get rotated independent of each other, maximum angel by witch the entire paragraph gets rotated)
    :return:
        trainingsdata: list of (grayscale imag, goldlabel text)-tupels
    """
    #TODO check if dir exists and contains the correct data.
    if img_type not in [vars(ImgTypes)[x] for x in vars(ImgTypes).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_loader, ", ", img_type, ", ", maxcount, "): invalid input, img_type should be img_types.*")
        return None
    if dataset_loader not in [vars(DatasetNames)[x] for x in vars(DatasetNames).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_loader, ", ", img_type, ", ", maxcount, "): invalid input, dataset should be dataset_names.iam")
        return None
    if goldlabel_type not in [vars(GoldlabelTypes)[x] for x in vars(GoldlabelTypes).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_loader, ", ", img_type, ", ", maxcount, "): invalid input, goldlabel_type should be ", GoldlabelTypes)
        return None
    # <should_be_parameters_buts_its_more_convenient_to_have_them_here>
    words_per_line = [2, 3]  # number of words per line
    lines_per_paragrph = [3, 4, 5]  # number of lines per paragraph

    word_distance = [10, 20]  # padding added left of each word
    line_distance = [5, 10]  # padding added upward of each line

    max_chars_per_line = 64  # TODO enforce this limit when building lines
    # </should_be_parameters_buts_its_more_convenient_to_have_them_here>

    if dataset_loader == DatasetNames.iam:
        data = load_iam(dir, goldlabel_type)  # [(relative path of img file, goldlabel text of that file)]
    else:
        raise "invalid dataset_loader: "+str(dataset_loader)
    #print("path_gl: ", getType(data))
    #print("data_imgpath_goldlabel = ", data[:5])
    # if goldlabel_type == text: type(data) = [(img: np.array(?, ?), text: string)]
    # if goldlabel_type == linepositions: type(data) = [(img: np.array(?, ?), linepoint: [1:point: (int, int)])]

    if 0 < maxcount < len(data) and offset+maxcount < len(data):
        if img_type == ImgTypes.word:
            maxcount = maxcount
        elif img_type == ImgTypes.line:
            maxcount = int(maxcount*max(words_per_line))
        elif img_type == ImgTypes.paragraph:
            maxcount = int(maxcount*max(words_per_line)*max(lines_per_paragrph))
        else:
            print("unexpected img_type: ", img_type)
            return None
        data = data[offset:offset+maxcount]
    #print("path_gl_short: ", getType(data))
    if goldlabel_type == GoldlabelTypes.lineimg:
        data = [(path, load_img(dir+"/"+path)) for (path, gl) in data]
    data = [(load_img(dir+"/"+path), gl) for (path, gl) in data]
    #print("imgword_gl: ", getType(data))
    if img_type == ImgTypes.word:
        if goldlabel_type == GoldlabelTypes.linepositions:
            ys = 1
        elif goldlabel_type == GoldlabelTypes.text:
            ys = x_size[1]//4  # //4 weil konstante in Models.simpleHTR # max([len(d[1]) for d in data])
        elif goldlabel_type == GoldlabelTypes.lineimg:
            ys = (32, 256)  # should be a parameter or something
        else:
            ys = 1
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=ys)
        print("imgwordenc_gl: ", getType(data))
        return data

    # data [(word-img, image of the same word)] still correct
    tmp = [data[i:i+random.choice(words_per_line)] for i in range(0, len(data), max(words_per_line))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=1, pad=[random.choice(word_distance) for _ in range(len(t))]) for t in tmp]
    data = [rotate_img(img, gl, goldlabel_type, angle=random.randrange(-line_para_winkel[0], line_para_winkel[0]+1, 1)) for (img, gl) in data]
    #print("data_lines[0]: ", data[0])
    #print("data_imgline_goldlabel = ", data[:5])
    #print("imgline_gl: ", getType(data))
    if img_type == ImgTypes.line:  # line
        if goldlabel_type == GoldlabelTypes.linepositions:
            ys = 1
        elif goldlabel_type == GoldlabelTypes.text:
            ys = max_chars_per_line  # max([len(d[1]) for d in data])
        else:
            ys = 1
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=ys)
        #print("imglineenc_gl: ", getType(data))
        return data
    # rotate lines
    # still correct lienimg
    tmp = [data[i:i+random.choice(lines_per_paragrph)] for i in range(0, len(data), max(lines_per_paragrph))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=0, pad=[random.choice(line_distance) for unused in range(len(t))]) for t in tmp]
    data = [rotate_img(img, lp, goldlabel_type, angle=random.randrange(-line_para_winkel[1], line_para_winkel[1]+1, 1)) for (img, lp) in data]
    #print("imgpara_gl: ", getType(data))
    # still correct lineimg
    #print("data_parag[0]: ", data[0])
    #print("data_imgpara_goldlabel = ", data[:5])
    if img_type == ImgTypes.paragraph:  # paragraph
        if goldlabel_type == GoldlabelTypes.linepositions:
            ys = max(lines_per_paragrph)
        elif goldlabel_type == GoldlabelTypes.text:
            ys = max_chars_per_line*max(lines_per_paragrph)  # max([len(d[1]) for d in data])
        else:
            ys = 1
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=ys)
        #print("Dataloader.getData: encoded para: ", getType(data))
        return data
    return "Dataloader.getData: This return statement is impossible to reach."


class abstractDataset:
    name = "TODO overwrite in subclasses"
    x = [float]
    y = [float]
    def get_batch(self, n:int) -> ([x], [y]):
        """
        :param n:
        the number of elements in the batch returned
        :return:
        (data, goldlabel), so that tf.keras.model.fit(x=data, y=goldlabel, [...]) works.
        e.g. goldlabel[i] is the goldlabel for data[i]
        """
        return None

    def show(self, batch: ([x], [y]), predicted: [y] = None) -> None:
        """
        :param batch:
        a batch of the same type as returned by get_batch
        :param predicted:
        if not None should be the same type and dimensions as batch[1]
        :return:
        None, but displays the batch. optionally with predicted labels
        """
        return None


class Dataset_test(abstractDataset):
    name = "test"
    def __init__(self, difficulty=0):
        self.diff = difficulty
        self.name = "test"+str(self.diff)

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
            if self.diff < 4:
                pos.sort(key=lambda x: x[0])
                r.append((img, points2dense(pos)))
        return np.array([d[0] for d in r]), np.array([d[1] for d in r])  # img, goldlabel

    def show(self, batch, predicted=None):
        """
        :param batch:
        :type ([img], [goldlabels])
        with img is type as in opencCV
        and goldlabel is [float]
        e.g. the return value of self.get_batch:
        x, y = dataset.get_batch(20)
        dataset.show((x, y))
        :return:
        None, but shows the batch and optional predictions
        """
        assert len(batch[0]) == len(batch[1])
        batch = [(np.array(batch[0][i], dtype="uint8"), dense2points(batch[1][i])) for i in range(len(batch[0]))]
        for i in range(len(batch)):
            #print("test_dataset.show: img: "+str(img)+" -> "+str(gl))
            (img, gl) = batch[i]
            for point in gl:
                cv2.circle(img, point, radius=5, color=0, thickness=3)
            if predicted is not None:
                for p in dense2points(predicted[i]):
                    cv2.circle(img, p, radius=5, color=125, thickness=2)
            cv2.imshow("todo", img)
            cv2.waitKey(0)

class RNNDataset(abstractDataset):
    name = "RNN test dataset"
    def get_batch(self, size):
        x_train = []
        y_train = []
        for i in range(size):
            nTimestams = random.choice([4])
            val = random.choice([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
            x_tmp = [val]+[(0, 0, 0)]*(nTimestams-1)
            y_tmp = [val]*nTimestams
            x_train.append(x_tmp)
            y_train.append(y_tmp)
        return np.array(x_train), np.array(y_train)

    def show(self, batch, predicted=None):
        print("X -> Y| GL")
        if predicted==None:
            predicted = [""]*len(batch[0])
        for i in range(len(batch[0])):
            print(batch[0][i], "->", batch[1][i], "| ", predicted[i])

class img2lineimgDataset(abstractDataset):
    name = "img2lineimgDatset"
    x = [[float]]
    y = [[[float]]]

    def __init__(self):
        self.imgsize = (512, 1024)
        self.lineszie = (32, 256)

    def get_batch(self, n: int) -> ([x], [y]):
        # type(batch) = (x:(n, w, h), y:(max_linecount, n, linesize[0], linesize[1]))
        data = getData(dir=data_dir, dataset_loader=DatasetNames.iam, img_type=ImgTypes.paragraph, goldlabel_type=GoldlabelTypes.lineimg, goldlabel_encoding=GoldlabelEncodings.dense, maxcount=n, line_para_winkel=(0, 0), x_size=self.imgsize)
        x = []
        y = [[] for _ in range(max([len(gl) for (img, gl) in data]))]
        for (img, gl) in data:
            x.append(img)
            while len(gl) < len(y):
                gl.append(np.zeros(self.lineszie))
            for i in range(len(y)):
                #print("Dataloader.img2lineimgDataset: gl[i].shape = ", gl[i].shape)
                y[i].append(gl[i])
        x_train = np.array(x)
        y_train = [np.array(yn) for yn in y]
        return x_train, y_train

    def show(self, batch: ([x], [y]), predicted: [y] = None) -> None:
        print("Dataloader.img2lineimgDataset: show batch = ", getType(batch), "\n pred = ", getType(predicted))
        for batch_i in range(len(batch[0])):
            (img, gl) = (batch[0][batch_i], [batch[1][line][batch_i] for line in range(len(batch[1]))])
            img = np.array(img, dtype="uint8")
            gl = [np.array(li, dtype="uint8") for li in gl]
            cv2.imshow("image: ", img)
            for gl_i in range(len(gl)):
                cv2.imshow("lineimg:"+str(gl_i), gl[gl_i])
            if predicted is not None:
                for pred_i in range(len(predicted[batch_i])):
                    pred = predicted[batch_i][pred_i]
                    #pred = np.reshape(pred, (pred.shape[1], pred.shape[2]))
                    pred = np.array([[max(min(255, t), 0)for t in tmp] for tmp in pred], dtype="uint8")
                    cv2.imshow("predlineimg:"+str(pred_i), pred)

            cv2.waitKey(0)


class Dataset(abstractDataset):
    name = "real"
    data_directory = None
    gl_type = 0
    pos = 0
    dataset_size = -1
    imgsize = {ImgTypes.word: (32, 64), ImgTypes.line: (32, 256), ImgTypes.paragraph: (512, 1024)}
    # TODO add glsize or similar
    gl_encoding = {GoldlabelTypes.text: GoldlabelEncodings.onehot, GoldlabelTypes.linepositions: GoldlabelEncodings.dense, GoldlabelTypes.lineimg: GoldlabelEncodings.dense, GoldlabelTypes.number_of_lines: GoldlabelEncodings.dense}
    add_empty_images = True
    #super.x = [[float]]  # image
    #super.y = [float]  # linepoint

    def __init__(self, img_type, gl_type):
        self.data_directory = data_dir
        self.img_type = img_type
        self.gl_type = gl_type

        self.line_para_winkel = (3, 5)
        self.do_not_fix_dimensions_just_flip = False

        self.gl_encoding = self.gl_encoding[self.gl_type]
        self.imgsize = self.imgsize[self.img_type]

        self.pos = 0
        self.dataset_size = len(load_iam(self.data_directory, gl_type))



    def get_batch(self, size):
        size = size-3  # drei leere bilder hunzugefuegt
        #selekt witch part of dset to use
        assert size < self.dataset_size
        if self.pos+size >= self.dataset_size:
            self.pos = self.pos % (self.dataset_size-size)
        data = getData(dir=self.data_directory, dataset_loader=DatasetNames.iam, img_type=self.img_type, goldlabel_type=self.gl_type, goldlabel_encoding=self.gl_encoding, maxcount=size, line_para_winkel=self.line_para_winkel, x_size=self.imgsize)
        self.imgsize = data[0][0].shape
        self.pos = self.pos+size
        # include empty images
        if self.add_empty_images:
            yshape = np.array(data[0][1]).shape  # data[0][1] might already be a np-array
            img = np.zeros(shape=self.imgsize, dtype="uint8")
            #gl = np.array([0]*len(data[0][1]))
            gl = np.zeros(yshape)
            data.append((img, gl))

            img = np.full(shape=self.imgsize, fill_value=255, dtype="uint8")
            #gl = np.array([0]*len(data[0][1]))
            gl = np.zeros(yshape)
            data.append((img, gl))

            img = np.array([[random.randrange(0, 255, 1) for unused in range(self.imgsize[1])] for unused in range(self.imgsize[0])])
            #gl = np.array([0]*len(data[0][1]))
            gl = np.zeros(yshape)
            data.append((img, gl))
        print("Dataloader.Dataset.get_batch: data = ", getType(data))
        if self.do_not_fix_dimensions_just_flip:
            x_train = np.array([np.transpose(d[0]) for d in data], dtype=float)
        else:
            x_train = np.array([d[0] for d in data], dtype=float)
        y_train = np.array([d[1] for d in data], dtype=float)
        #return tf.data.Dataset.from_tensor_slices((x_train, y_train))  # maybe using this could be more efficent
        return x_train, y_train


    def show(self, batch, predicted=None):
        print("Dataloader.Dataset.show: start")
        assert len(batch[0]) == len(batch[1])
        batch = [(batch[0][i], batch[1][i]) for i in range(len(batch[0]))]
        for i in range(len(batch)):
            (img, gl) = batch[i]
            img = np.array(img, dtype="uint8")
            #print("show_points_data.img: ", getType(img))
            #print("show_points_data.pts: ", getType(points))
            #print("show_points_data.pts: ", points)
            if self.gl_type == GoldlabelTypes.linepositions:
                h, w = batch[0][0].shape
                if self.gl_encoding == GoldlabelEncodings.dense:
                    decode_func = dense2linepoints
                elif self.gl_encoding == GoldlabelEncodings.onehot:
                    decode_func = sparse2linepoints
                else:
                    print("TODO unsupported gl_encoding in dataset.show: ", self.gl_encoding)
                    return None
                points = decode_func(gl, max_x=w, max_y=h)
                predPoints = []
                if predicted is not None:
                    predPoints = decode_func(predicted[i], max_x=w, max_y=h)
                for point in points:
                    #print("point = ", point)
                    #point = ((max(1, int(point[0][0])), max(1, int(point[0][1]))), (max(1, int(point[1][0])), max(1, int(point[1][1]))), max(1, int(point[2])))
                    cv2.circle(img, point[0], int(point[2]/2), 0, 3)
                    cv2.rectangle(img, (point[0][0]-5, point[0][1]-5), (point[0][0]+5, point[0][1]+5), 0, 2)
                    cv2.circle(img, point[1], int(point[2]/2), 0, 3)
                    cv2.line(img, point[0], point[1], 0, thickness=3)
                for point in predPoints:
                    cv2.circle(img, point[0], radius=int(point[2]/2), color=125, thickness=2)
                    cv2.rectangle(img, (point[0][0]-5, point[0][1]-5), (point[0][0]+5, point[0][1]+5), 125, 1)
                    cv2.circle(img, point[1], int(point[2]/2), 125, 2)
                    cv2.line(img, point[0], point[1], 125, thickness=2)
            elif self.gl_type == GoldlabelTypes.text:
                if self.gl_encoding == GoldlabelEncodings.dense:
                    decode_func = dense2txt
                elif self.gl_encoding == GoldlabelEncodings.onehot:
                    decode_func = sparse2txt
                else:
                    print("TODO unsupported gl_encoding in dataset.show: ", self.gl_encoding)
                    return None
                gl = decode_func(gl, y_size=32)
                if predicted is not None:
                    print("Dataloader.Dataset: pred = ", getType(predicted[i]))
                    print("pred = ", decode_func(predicted[i], y_size=32))
                print("gl = ", gl)
            cv2.imshow(str(gl), img)
            cv2.waitKey(0)


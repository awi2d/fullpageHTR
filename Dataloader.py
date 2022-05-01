import cv2
import numpy as np
import tensorflow as tf
import random
# import tensorflow_datasets as tf_ds
# TODO https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# TODO tf.keras.utils.image_dataset_from_directory nutzen
# 125 GB RAM auf server (gesamt)

line_point = ((int, int), (int, int), int)  # (startpoint of line, endpoint of line, height)
linepoint_length = 5

data_dir = "../SimpleHTR/data/trainingDataset/"  # The dirctory that is mapped to not be in the docker
#data_dir = "C:/Users/Idefix/PycharmProjects/SimpleHTR/data/"
iam_dir = data_dir + "iam/"  # the unchanged iam dataset
dataset_dir = data_dir + "generated/"  # directoy for storing generated data
models_dir = data_dir + "models/"  # directoy for storing trained models


class GoldlabelTypes:
    text = 0  # str the text that is written in the image, e.g. "A move to Stop mr. Gaitskell from nominating any more labour life peers is to be made at a meeting"
    linepositions = 1  # (start_of_line:(x:int, y:int), end_of_line:(x2:int, y2:int), height_of_line:int) all information needed to manually segment the line out of the image
    number_of_lines = 2  # int
    lineimg = 4  # list of all textlines, each with the same size (32x256), only supports dense encoding
    # maybe add new type: linemask: [image where exactly the pixels of line i are 1, everything else 0 for i in range(max(lines_per_paragrph))]
    # attention layer maybe supports masking
    # masking may only support RNN and only one bool per timestamp https://www.tensorflow.org/guide/keras/custom_layers_and_models
    # image segmentation (NN that predicts label for each pixel) https://www.tensorflow.org/tutorials/images/segmentation


class GoldlabelEncodings:
    onehot = 10
    dense = 11


class ImgTypes:
    word = 20  # the image contains a single word
    line = 21  # the image contains multiple words concatenated horizontally
    paragraph = 22  # the image contains multiple lines concatenated vertically


class DatasetNames:
    iam = 30


def extractline(img, linepoint: [float], y_size = (32, 256)):
    """
    :param img:
    :param linepoint:
    dense encoding of one linepoint
    :param y_size:
    (height of lineimg, widht of lineimg)
    dimensions of the image returned by this function.
    :return:
    the normalised part of the image where the text line described by linepoint should be
    """
    assert len(linepoint) == 5
    img = np.array(img, dtype="uint8")

    #print("Datloader.extractline: linepoint = ", linepoint)
    (h_img, w) = img.shape
    #print("Dataloader.extractline: img.shape_prae = ", (h_img, w))
    linepoint = dense2linepoints(linepoint, x_size=(h_img, w))
    #print("Datloader.extractline: linepoint = ", linepoint)
    #rotate image so that line is horizontal
    ((x1, y1), (x2, y2), h_lp) = linepoint[0]
    h_lp = h_lp//2
    #print("Dataloader.extractline: linepoint[0] = ", linepoint[0])
    # y1 == y2 => ist bereits gerade
    if abs(x2-x1)+abs(y2-y1) < 5 or h_lp < 2:
        # <=> is empty  ((0, 0), (0, 0), 0) linepoint
        print("Dataloader.extractline: empty linepoint -> return black empty image")
        return np.zeros((32, 256))
    if x1 < 0 or x1 > w or x2 < 0 or x2 > w or y1 < 0 or y1 > h_img or y2 < 0 or y2 > h_img:
        print("Dataloader.extractline: linepoint out of image -> return black empty image")
        return np.zeros((32, 256))
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
    #print("Dataloader.extractline: linepoint_rotated = ", ((x1, y1), (x2, y2), h_lp))
    (h_img, w) = img.shape
    #print("Dataloader.extractline: img.shape_rot = ", (h_img, w))
    y = int(0.5*y1+0.5*y2)  # y1 and y2 should already be almost the same
    #left_bound = max(0, int(x1-h_lp))
    #right_bound = min(h_img, int(x2+h_lp))
    #upper_bound = max(0, int(y-h_lp))
    #lower_bound = min(w, int(y+h_lp))
    left_bound = max(0, int(x1-h_lp))
    right_bound = max(0, int(x2+h_lp))
    upper_bound = max(0, int(y-h_lp))
    lower_bound = max(0, int(y+h_lp))
    #print("Dataloader.extractline: bounds = ", ((left_bound, right_bound), (upper_bound, lower_bound)))
    #a = cv2.line(img, (x1, y1), (x2, y2), 125, thickness=3)
    #a = cv2.rectangle(a, (left_bound, upper_bound), (right_bound, lower_bound), 125)
    #cv2.imshow("a", a)
    #cv2.waitKey(0)
    #img = np.array(img[left_bound:right_bound, upper_bound:lower_bound], dtype="uint8")
    img = np.array(img[upper_bound:lower_bound, left_bound:right_bound], dtype="uint8")
    #print("Dataloader.extractline: img.shape_cutted = ", img.shape)

    # make img have correct shape without distorting it or losing information. maybe make this into seperate method
    fitshape_then_cut(img, y_size)
    return img


# <debug functions>

def getType(x):
    """
    :param x:
    :return:
    a string containing information about the type of x
    """
    deliminating_chars = {"list": ('[', ';', ']'), "tupel": ('<', ';', '>'), "dict": ('{', ';', '}')}
    name = type(x).__name__
    if name == 'list':
        return '['+str(len(x))+":"+getType(x[0])+']'  # assumes all element of the list have the same type
    if name == "str":
        return "str_"+str(len(x))
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
    if name == "Dataset":
        return "Dataset(name=\""+str(x.name)+"\", "+str(x.imgsize)+"->"+str(x.glsize)+" )"
    if name in ["KerasTensor", "Tensor", "EagerTensor"]:
        return str(name)+"("+str(x.shape)+":"+str(x.dtype)+")"
    return name


def showData(data, gltype=GoldlabelTypes.linepositions, encoded=False):
    """
    :param data:
    data in the [(img, gl-unencoded)]-format.
    as used in getData
    :param gltype:
    :return:
    shows the data
    """
    print("Dataloader.showData(gltype=", gltype, ")")
    if gltype == GoldlabelTypes.linepositions:
        for i in range(len(data)):
            print("Dataloader.showData: data["+str(i)+"].gl = ", data[i][1])
            img = np.array(data[i][0], dtype="uint8")
            if encoded:
                gl_list = dense2linepoints(data[i][1], x_size=(32, 64))
            else:
                gl_list = data[i][1]
            for gl in gl_list:
                img = cv2.line(img, gl[0], gl[1], color=125)
                img = cv2.circle(img, gl[0], radius=int(gl[2]/2), color=125, thickness=3)
                img = cv2.circle(img, gl[1], radius=int(gl[2]/2), color=125, thickness=3)
            cv2.imshow("Dataloader.showData:"+str(i), img)
            cv2.waitKey(0)
    elif gltype == GoldlabelTypes.lineimg:
        for (img, lineimgs) in data:
            img = np.array(img, dtype="uint8")
            cv2.imshow("Dataloader.showData: img", img)
            for ili in range(len(lineimgs)):
                cv2.imshow("Dataloader.showData: line"+str(ili), np.array(lineimgs[ili], dtype="uint8"))
            cv2.waitKey(0)
    else:
        print("Dataloader.showData: TODO implement for gltype: ", gltype)


def test_tfds():
    # unused function.
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

# <encoding/decoding functions>

def num2sparse(a:int, max):
    r = np.zeros(max)
    r[a] = 1
    return r
def sparse2num(a):
    return np.argmax(a)


alphabet = np.array(["-", " "]+list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMOPQRSTUVWXYZ1234567890.,:;!#"))  # "-" is ctc blank label, "#" is linebreak
def txt2sparse(txt, y_size:(int, int)):
    """
    :param txt:
    :param y_size:
    (maximum length of text, number of different chars = len(alphabet))
    :return:
    sparse representation of txt, padded with ctc blank label. can be reconverted to text by sparse2txt.
    """
    #print("Dataloader.txt2sparse: txt ", len(txt), ", ysize ", y_size)
    assert len(txt) <= y_size[0]
    assert y_size[1] == len(alphabet)
    #txt = txt.lower()
    a = np.array([(np.argwhere(alphabet == c)[0][0] if c in alphabet else 1) for c in txt])
    #b = np.zeros(y_size*len(alphabet))
    b = np.zeros(y_size)
    for i in range(len(a)):
        #b[i*len(alphabet)+a[i]] = 1
        b[i][a[i]] = 1
    return np.array(b)
def sparse2txt(sparse):
    dense = [str(alphabet[np.argmax(sparse_char)]) for sparse_char in sparse]
    return "".join(dense)


def txt2dense(txt, y_size: int):
    """
    :param txt:
    :param y_size:
    maximum length of text
    :return:
    dense representation of txt, padded with ctc blank label. can be reconverted to text by dense2txt.
    """
    assert len(txt) <= y_size
    return np.array([(np.argwhere(alphabet == c)[0][0] if c in alphabet else 0) for c in (txt.lower()+"-"*(y_size[0]-len(txt)))])
def dense2txt(txt):
    return "".join([str(alphabet[t]) for t in txt])


def linepoint2sparse(points: [line_point], x_size:(int, int), y_size) -> [float]:
    assert max([max(point[0][0], point[1][0]) for point in points]) < x_size[0]
    assert max([max(point[0][1], point[1][1], point[2]) for point in points]) < x_size[1]
    assert y_size >= len(points)
    point_length = 2 * x_size[0] + 3 * x_size[1]  # number of bits needed to encode one point
    r = [0]*point_length*y_size
    #r[s*point_length+0:s*point_length+max_x] = one-hot encoding von points[s][0][0]
    for i in range(len(points)):
        ((xs, ys), (xe, ye), h) = points[i]
        offset = point_length*i
        r[offset+int(xs)] = 1
        r[offset + x_size[0] + int(ys)] = 1
        r[offset + x_size[0] + x_size[1] + int(xe)] = 1
        r[offset + 2 * x_size[0] + x_size[1] + int(ye)] = 1
        r[offset + 2 * x_size[0] + 2 * x_size[1] + int(h)] = 1
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


def linepoint2dense(points: [line_point], x_size: (int, int), y_size: int) -> [float]:
    assert max([max(point[0][0], point[1][0]) for point in points]) <= x_size[1]
    assert max([max(point[0][1], point[1][1], point[2]) for point in points]) <= x_size[0]
    #print("Dataloader.linepoint2dense: y_size = ", y_size)
    assert y_size >= linepoint_length*len(points)
    dp = [(xs/x_size[0], ys/x_size[1], xe/x_size[0], ye/x_size[1], h/x_size[1]) for ((xs, ys), (xe, ye), h) in points]
    dp = list(sum(dp, ()))  # flattens the list. https://stackoverflow.com/questions/10632839/transform-list-of-tuples-into-a-flat-list-or-a-matrix/35228431
    assert len(dp) % linepoint_length == 0
    return np.array(dp+[0]*(y_size-len(dp)))
def dense2linepoints(points: [float], x_size:(int, int)) -> [line_point]:
    assert len(points) % linepoint_length == 0
    return [((int(points[i]*x_size[0]), int(points[i+1]*x_size[1])), (int(points[i+2]*x_size[0]), int(points[i+3]*x_size[1])), max(1, int(points[i+4]*x_size[1]))) for i in range(0, len(points), 5)]


def lineimgs2dense(lineimgs, y_size):
    return [fitshape_then_cut(li, y_size) for li in lineimgs]
    #return [cv2.resize(li, (y_size[1], y_size[0])) for li in lineimgs]
def dense2lineimgs(lineimgs):
    return lineimgs
# </encoding/decoding functions>


def fitshape_then_cut(img, shape):
    """
    scales the img to be smaller than shape without distorting it, then pads img to correct size.
    :param img:
    an image
    :param shape:
    the shape the image will have after this operation
    :return:
    """
    y_size = shape
    if img.shape[0]/y_size[0] >= img.shape[1]/y_size[1]:
        img = cv2.resize(img, (int(img.shape[1] * (y_size[0]/img.shape[0])), y_size[0]), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (y_size[1], int(img.shape[0] * (y_size[1]/img.shape[1]))), interpolation=cv2.INTER_AREA)
    img = np.pad(img, ((0, max(0, y_size[0]-img.shape[0])), (0, max(0, y_size[1]-img.shape[1]))), mode='constant', constant_values=255)
    img = np.array(img[0:y_size[0], 0:y_size[1]], dtype="uint8")
    assert (img.shape[0], img.shape[1]) == (shape[0], shape[1])
    return img

def scale(img, goldlabel, x_size, gl_type):
    """
    wrapper around cv2.resize to keep img and goldlabel consistent
    :param img:
    :param goldlabel:
    :param x_size:
    size the img should have after scale.
    None means to scale that axis by the same amount as the other one.
    :param gl_type:
    :return:
    the image and goldlabel, so that they still are consistent and the image has exactly the shape x_size
    """
    if x_size == (None, None):
        return np.array(img, dtype="uint8"), goldlabel

    if x_size[0] is None:
        x_size = (int(img.shape[0] * (x_size[1]/img.shape[1])), x_size[1])
    if x_size[1] is None:
        x_size = (x_size[0], int(img.shape[1] * (x_size[0]/img.shape[0])))
    if gl_type == GoldlabelTypes.linepositions:
        (h, w) = img.shape
        (nh, nw) = x_size
        fh = nh/h
        fw = nw/w
        goldlabel = [((int(x1*fw), int(y1*fh)), (int(x2*fw), int(y2*fh)), int(h_lp*fh)) for ((x1, y1), (x2, y2), h_lp) in goldlabel]
    # linepoints and text do not change
    img = cv2.resize(img, (x_size[1], x_size[0]))
    assert img.shape == x_size
    return np.array(img, dtype="uint8"), goldlabel


def cut(img, goldlabel, x_size, gl_type=GoldlabelTypes):
    """
    wrapper around slicing/padding operation to keep img and goldlabel consisten
    :return:
    the image and goldlabel, so that they still are consistent, unscaled and the image has exactly the shape (h, w)
    """
    #print("Dataloader.fitsiz: (h, w) = ", (h, w))
    if gl_type == GoldlabelTypes.linepositions:
        # remove linepoints that are outside the cutted area
        goldlabel = [((x1, y1), (x2, y2), h_lp) if (max(y1, y2)+h_lp//2) < x_size[0]+5 else ((0, 0), (0, 0), 0) for ((x1, y1), (x2, y2), h_lp) in goldlabel]
        # cut linepoints that are partially outside the cutted area
        goldlabel = [((min(x1, x_size[1]-h_lp), y1), (min(x2, x_size[1]-h_lp), y2), h_lp) for ((x1, y1), (x2, y2), h_lp) in goldlabel]
    elif gl_type == GoldlabelTypes.text:
        goldlabel = goldlabel  # TODO text that is cutted out of the image should be cut out of the goldlabel
    elif gl_type == GoldlabelTypes.lineimg:
        goldlabel = goldlabel  # TODO if line is partially cutted from image lineimg should be removed from goldlabel
    else:
        print("Invalid goldlabelencoding in Dataloader.fitsize: "+str(gl_type))
        raise "Invalid goldlabelencoding in Dataloader.fitsize: "+str(gl_type)
    img = np.pad(img, ((0, max(0, x_size[0]-img.shape[0])), (0, max(0, x_size[1]-img.shape[1]))), mode='constant', constant_values=255)
    img = np.array(img[0:x_size[0], 0:x_size[1]], dtype="uint8")
    #print("fitsize: (h, w) = ", (h, w), "shape = ", img.shape)
    assert img.shape == x_size
    return img, goldlabel


def encode_and_pad(data, goldlabel_type, goldlabel_encoding, x_size=None, y_size=None):
    """
    :param data:
    data of type [(image, goldlabel)]
    :param goldlabel_type:
    type of goldlabel in Dataset.gl_type
    currently text or lineposition
    :param goldlabel_encoding:
    encoding of goldlabel in Dataset.gl_encoding
    :param x_size:
    (width: int, height: int) or the other way round, size all images in data will have.
    that means they either get padded or downscaled and clipped.
    :param y_size:
    size all goldlabel_encodings in data will have
    :return:
    [(image, goldlabel)] like input, but the goldlabel is encoded and image and goldlabel are padded to size
    """

    if x_size is None:
        h = int(max([img.shape[0] for (img, gl) in data]))
        w = int(max([img.shape[1] for (img, gl) in data]))
        #print("Dataloader.encode_and_pad: shape in w, h = ", h, ", ", w)
        x_size = (h, w)
    if y_size is None:
        gl_shapes = [np.array(gl).shape for (img, gl) in data]
        y_size = [max([shape[i] for shape in gl_shapes]) for i in range(len(gl_shapes[0]))]  # assumes that all goldlabes have the same number of dimensions
        if len(y_size) == 1:
            y_size = y_size[0]  # maybe a bad idea
    #print("Dataloader.encode_and_pad: used x_size = (w, h) = ", (w, h))

    # rescale to correct height, then cut/pad to correct width
    #data = [scale(img, gl, x_size=(x_size[0], None), gl_type=goldlabel_type) for (img, gl) in data]
    data = [cut(img, gl, x_size=x_size, gl_type=goldlabel_type) for (img, gl) in data]
    #print("Dataloader.encode_and_pad: aftercut_data.gl = ", [gl for (img, gl) in data])
    if goldlabel_type == GoldlabelTypes.text:
        if goldlabel_encoding == GoldlabelEncodings.dense:
            print("dense text encoding is currently unsupported, return text in UTF-8 as goldlabel instead")
            return [(img, txt2dense(txt, y_size)) for (img, txt) in data]  # padding should already be met by fitsize
        elif goldlabel_encoding == GoldlabelEncodings.onehot:  # text in one-hot
            return [(img, txt2sparse(txt, y_size)) for (img, txt) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == GoldlabelTypes.linepositions:
        if goldlabel_encoding == GoldlabelEncodings.dense:
            data = [(img, linepoint2dense(point, x_size=x_size, y_size=y_size)) for (img, point) in data]
            return data
        elif goldlabel_encoding == GoldlabelEncodings.onehot:
            return [(img, linepoint2sparse(point, x_size=x_size, y_size=y_size)) for (img, point) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == GoldlabelTypes.number_of_lines:
        maxlinecount = max([gl for (img, gl) in data])
        if goldlabel_encoding == GoldlabelEncodings.dense:
            return [(img, lc/maxlinecount) for (img, lc) in data]
        elif goldlabel_encoding == GoldlabelEncodings.onehot:
            return [(img, num2sparse(lc, maxlinecount)) for (img, lc) in data]
        else:
            raise "invalid goldlabel_encoding: "+str(goldlabel_encoding)
    elif goldlabel_type == GoldlabelTypes.lineimg:
        # does not support onehot encoding, uses dense in any case
        max_num_li = max([len(lineimgs) for (img, lineimgs) in data])
        # add empty lineimgs to pad to same size
        data = [(img, lineimgs+[np.zeros(y_size, dtype="uint8")]*(max_num_li-len(lineimgs))) for (img, lineimgs) in data]
        return [(img, lineimgs2dense(lineimgs, y_size=y_size)) for (img, lineimgs) in data]
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
        # np.pad(np-array, ((oben, unten), (links, rechts))
        img_list = [np.pad(img_list[i], ((pad[i], 0), (0, w-img_list[i].shape[1])), mode='constant', constant_values=255) for i in range(len(img_list))]  # white padding added at right margin
        if goldlabel_type == GoldlabelTypes.text:
            goldlabel = ''.join(goldlabel_list)
        elif goldlabel_type == GoldlabelTypes.linepositions:
            goldlabel = [((0, 0), (0, 0), 0)]*len(goldlabel_list)
            offset = 0
            for i_gl in range(len(goldlabel_list)):  # gl is list of points
                #print("goldlabe_list[", i_gl, "] = ", str(goldlabel_list[i_gl]))
                pre_gl = goldlabel_list[i_gl][0]  # (startpoint, endpoint, height)
                goldlabel[i_gl] = ((pre_gl[0][0], pre_gl[0][1]+offset+pad[i_gl]), (pre_gl[1][0], pre_gl[1][1]+offset+pad[i_gl]), pre_gl[2])
                #offset += pad[i_gl]+pre_gl[2]
                offset += img_list[i_gl].shape[0]
        elif goldlabel_type == GoldlabelTypes.lineimg:
            # axis == 0 -> bilder untereinander -> img_list is list of lineimg
            goldlabel = [lineimgs[0] for lineimgs in goldlabel_list]  # assume that each line being concated only has one lineimg
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
            print("Dataloader.concat_data: gl_list shapes = ", getType(goldlabel_list))
            mh = max([max([limg.shape[0] for limg in lineimgs]) for lineimgs in goldlabel_list])
            #print("Dataloader.concat_data: gl_list shapes = ", [gl.shape for gl in goldlabel_list])
            goldlabel_list = [[np.pad(limg, ((0, mh-limg.shape[0]), (0, 0)), mode='constant', constant_values=255) for limg in lineimgs] for lineimgs in goldlabel_list]
            #print("Dataloader.concat_data: gl_list = ", getType(goldlabel_list))
            #print("Dataloader.concat_data: gl_list = ", [gl.shape for gl in goldlabel_list])
            goldlabel = [np.concatenate([np.concatenate(lineimgs, axis=1) for lineimgs in goldlabel_list], axis=1)]
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
                goldlabel = img_filename
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
    image, gl = scale(image, gl, x_size=(h, w), gl_type=gl_type)
    return image, gl


def getData(dir, dataset_name=DatasetNames.iam, img_type=ImgTypes.paragraph, goldlabel_type=GoldlabelTypes.text,
            goldlabel_encoding=GoldlabelEncodings.onehot, x_size=(32, 256), y_size=5, maxcount=-1, offset=0,
            line_para_winkel=(2, 5), lines_per_paragrph=[3, 4, 5]):
    """
    :param dir:
    a directory that contains the dataset
    :param dataset_name:
    the Name of the dataset that should be loaded and edited. (currently only iam supported)
    :param img_type:
    element of ImgTypes. see there for documentation
    :param goldlabel_type:
    element of GoldlabelTypes. see there for documentation
    :param goldlabel_encoding:
    element of GoldlabelEncodings. see there for documentation
    :param x_size:
    np.array(x_value).shape == x_size will hold for all images in data returned by this function
    :param y_size:
    np.array(y_label).shape == y_size will hold for all y_label in data returned by this function
    note that this has to be consistent the shape required by max(lines_per_paragraph), gl_type and gl_encoding.
    e.g. with gl_type==linepoints and gl_encoding==dense, y_size has to be a scalar greater or equal then linepoint_length*max(lines_per_paragraph)
    :param maxcount:
    number of examples in the return data
    len(data)  == maxcount will hold.
    :param offset:
    from with wordimage in the iam-data to start
    :param line_para_winkel:
    all lines will get rotated by a random angel in degree or radiant in range(-line_para_winkel[0], line_para_winkel[0]+1)
    all paragraphs will get rotated by a random angel in degree or radiant in range(-line_para_winkel[1], line_para_winkel[1]+1)
    :param lines_per_paragrph:
    list of what numbers of lines per paragraph are possible
    :return:
    data to train a tf.model on, but zipped
    [(x_value, y_label)],
    where x_value is an image maybe concanated from images in dataset_name
    where y_label is the goldlabel as described in GoldlbaleTypes
    """
    #print("Dataloader.getData: ", "dir=", dir, "dataset_name=", dataset_name, "img_type=", img_type, "goldlabel_type=", goldlabel_type, "goldlabel_encoding=", goldlabel_encoding, "x_size=", x_size, "y_size=", y_size, "maxcount=", maxcount, "offset=", offset,"line_para_winkel=", line_para_winkel, "lines_per_paragrph=", lines_per_paragrph)
    #TODO check if dir exists and contains the correct data.
    if img_type not in [vars(ImgTypes)[x] for x in vars(ImgTypes).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_name, ", ", img_type, ", ", maxcount, "): invalid input, img_type should be img_types.*")
        return None
    if dataset_name not in [vars(DatasetNames)[x] for x in vars(DatasetNames).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_name, ", ", img_type, ", ", maxcount, "): invalid input, dataset should be dataset_names.iam")
        return None
    if goldlabel_type not in [vars(GoldlabelTypes)[x] for x in vars(GoldlabelTypes).keys() if not x.startswith("__")]:
        print("Dataloader.getData(", dir, ", ", dataset_name, ", ", img_type, ", ", maxcount, "): invalid input, goldlabel_type should be ", GoldlabelTypes)
        return None
    # <should_be_parameters_buts_its_more_convenient_to_have_them_here>
    words_per_line = [2, 3]  # number of words per line should be calculated from (wordimg_size after resizing to height) and y_size or max_chars_per_line

    word_distance = [10, 20]  # padding added left of each word
    line_distance = [5, 10]  # padding added upward of each line
    line_height = int(x_size[0]/max(lines_per_paragrph)-max(line_distance)-random.randint(0, 20))  # default height each word gets rescaled to before forming a line
    # line_height is only used in word, line and paragrph_img, NOT in lineimg_goldlabel

    # TODO y_size[0] when goldlabel_type==text (a.k.a. max_chars_per_line) limit should be obeyed when building lines,
    #  not lead to crash when encoding.
    # </should_be_parameters_buts_its_more_convenient_to_have_them_here>

    if dataset_name == DatasetNames.iam:
        data = load_iam(dir, goldlabel_type)  # getType(data) = [(relative path of wordimg file, goldlabel of that file)]
    else:
        raise "invalid dataset_loader: "+str(dataset_name)
    #print("path_gl: ", getType(data))
    #print("data_imgpath_goldlabel = ", data[:5])

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
        # remove unused datapoints before loading the images
        data = data[offset:offset+maxcount]
    #print("path_gl_short: ", getType(data))
    if goldlabel_type == GoldlabelTypes.lineimg:
        print("Dataloader.getData: y_size = ", y_size)
        data = [(path, load_img(dir+"/"+path)) for (path, gl) in data]  # load lineimg data
        data = [(path, [cv2.resize(limg, (int(limg.shape[1] * (y_size[0]/limg.shape[0])), y_size[0]))]) for (path, limg) in data]  # resize lineimg to correct height
    data = [(load_img(dir+"/"+path), gl) for (path, gl) in data]
    data = [scale(img=img, goldlabel=gl, x_size=(line_height, None), gl_type=goldlabel_type) for (img, gl) in data]
    #print("Dataloader.getData: imgword_gl: ", getType(data))

    # if goldlabel_type == text: type(data) = [(img: np.array(?, ?), text: string)]
    # if goldlabel_type == linepositions: type(data) = [(img: np.array(?, ?), linepoint: [1:linepoint: (start_of_line:(int, int), end_of_line:(int, int), height:int)])]
    # if goldlabel_type == lineimg: type(data) = [(img: np.array(?, ?), lineimg: the same as img)]

    if img_type == ImgTypes.word:
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, x_size=x_size, y_size=y_size)
        showData(data, encoded=True)
        print("Dataloader.getData: imgwordenc_gl: ", getType(data))
        return data

    # data [(word-img, image of the same word)]
    tmp = [data[i:i+random.choice(words_per_line)] for i in range(0, len(data), max(words_per_line))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=1, pad=[random.choice(word_distance) for _ in range(len(t))]) for t in tmp]
    data = [rotate_img(img, gl, goldlabel_type, angle=random.randrange(-line_para_winkel[0], line_para_winkel[0]+1, 1)) for (img, gl) in data]
    #print("data_lines[0]: ", data[0])
    #print("data_imgline_goldlabel = ", data[:5])
    #print("Dataloader.getData: imgline_gl: ", getType(data))
    if img_type == ImgTypes.line:  # line
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, x_size=x_size, y_size=y_size)
        print("Dataloader.getData: imglineenc_gl: ", getType(data))
        return data
    # rotate lines
    tmp = [data[i:i+random.choice(lines_per_paragrph)] for i in range(0, len(data), max(lines_per_paragrph))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=0, pad=[random.choice(line_distance) for unused in range(len(t))]) for t in tmp]
    data = [rotate_img(img, lp, goldlabel_type, angle=random.randrange(-line_para_winkel[1], line_para_winkel[1]+1, 1)) for (img, lp) in data]
    #print("imgpara_gl: ", getType(data))
    # still correct lineimg
    #print("data_parag[0]: ", data[0])
    #print("Dataloader.getData: imgpara_goldlabel = ", getType(data))
    if img_type == ImgTypes.paragraph:  # paragraph
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, x_size=x_size, y_size=y_size)
        #print("Dataloader.getData: imgparaenc_gl: ", getType(data))
        return data
    return "Dataloader.getData: This return statement is impossible to reach."


class abstractDataset:
    """
    interface to the trainingdata.
    this has nothing to do with tf.Data.Dataset
    example use:

    ds = Dataloader.Dataset(img_type=Dataloader.ImgTypes.xxx, gl_type=Dataloader.GoldlabelTypes.xxx)
    for i in range(10):
        (x, y) = ds.getBatch(64)
        tfkerasmodel.fit(x, y)

    or

    ds = Dataloader.Dataset(img_type=Dataloader.ImgTypes.xxx, gl_type=Dataloader.GoldlabelTypes.xxx)
    model = Models.xxx(input_shape=ds.imgsize, output_shape=ds.glsize)
    train(model, saveName="model_trained_with_"+ds.name, dataset=ds, batch_size=32)
    """
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


class RNNDataset(abstractDataset):
    # unused, I think
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


class Dataset(abstractDataset):
    """
    The interface to get training data.
    See abstractDataset for Documentation
    """
    name = "real"
    data_directory = None
    gl_type = 0
    pos = 0
    dataset_size = -1
    linecounts = [3, 4, 5]
    max_chars_per_line = 32
    imgsize = {ImgTypes.word: (32, 64), ImgTypes.line: (32, 256), ImgTypes.paragraph: (512, 1024)}
    # TODO wenn glsiz[linepositions] >= 100 (d.h. 20 zeilen) stürzt das Programm nach "Dataloader.Dataset.show: start" ohne Fehlermeldung mit \"Process finished with exit code -1073740791 (0xC0000409)\" ab.
    glsize = {GoldlabelTypes.text: None, GoldlabelTypes.linepositions: max(linecounts)*linepoint_length, GoldlabelTypes.lineimg: np.array((32, 256))}
    gl_encoding = {GoldlabelTypes.text: GoldlabelEncodings.onehot, GoldlabelTypes.linepositions: GoldlabelEncodings.dense, GoldlabelTypes.lineimg: GoldlabelEncodings.dense, GoldlabelTypes.number_of_lines: GoldlabelEncodings.dense}
    add_empty_images = True
    # typehints seem not to work well with inheritance
    #super.x = [[float]]  # image
    #super.y = [float]  # linepoint

    def __init__(self, img_type, gl_type):
        self.name = "Dataset_real("+str(img_type)+", "+str(gl_type)+")"
        self.data_directory = data_dir
        self.img_type = img_type
        self.gl_type = gl_type

        self.line_para_winkel = (3, 5)
        self.do_not_fix_dimensions_just_flip = False

        self.gl_encoding = self.gl_encoding[self.gl_type]
        self.imgsize = self.imgsize[self.img_type]
        self.glsize = self.glsize[self.gl_type]
        if self.gl_type == GoldlabelTypes.text:
            if self.img_type == ImgTypes.paragraph:
                self.glsize = np.array((self.max_chars_per_line*max(self.linecounts), len(alphabet)))
            else:
                self.glsize = np.array((self.max_chars_per_line, len(alphabet)))

        self.pos = 0
        self.dataset_size = len(load_iam(self.data_directory, gl_type))

    def get_batch(self, size):
        add_empty_images = False
        if size >= 6 and self.add_empty_images:
            add_empty_images = True
            size = size-3  # drei leere bilder hinzugefuegt
        #select witch part of dset to use
        assert size < self.dataset_size
        if self.pos+size >= self.dataset_size:
            self.pos = self.pos % (self.dataset_size-size)
        #print("Dataloader.Dataset.get_batch: self.glsize = ", self.glsize)
        data = getData(dir=self.data_directory, dataset_name=DatasetNames.iam, img_type=self.img_type, goldlabel_type=self.gl_type, goldlabel_encoding=self.gl_encoding, maxcount=size, line_para_winkel=self.line_para_winkel, x_size=self.imgsize, y_size=self.glsize, lines_per_paragrph=self.linecounts)
        # type of data: [(img, goldlabel)]
        #assert self.imgsize == data[0][0].shape  # assuming all images in data have the same size
        self.pos = self.pos+size
        # include empty images
        if add_empty_images:
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
            x_train = np.array([np.transpose(img) for (img, gl) in data], dtype=float)
        else:
            x_train = np.array([img for (img, gl) in data], dtype=float)
        #print("Dataloader.Dataset.get_batch: data.gl = ", [getType(gl) for (img, gl) in data])
        y_train = np.array([gl for (img, gl) in data], dtype=float)
        #return tf.data.Dataset.from_tensor_slices((x_train, y_train))  # maybe using this could be more efficent
        return x_train, y_train

    def show(self, batch, predicted=None):
        if type(batch) == int:
            batch = self.get_batch(batch)

        print("Dataloader.Dataset.show: start")
        assert len(batch[0]) == len(batch[1])
        batch = [(batch[0][i], batch[1][i]) for i in range(len(batch[0]))]
        for i in range(len(batch)):
            (img, gl) = batch[i]
            img = np.array(img, dtype="uint8")
            #print("Dataloader.Dataset.show: "+str(i)+" gl =", getType(gl))
            if self.gl_type == GoldlabelTypes.linepositions:
                h, w = batch[0][0].shape
                #assert (h, w) == self.imgsize  # should hold true, but dosnt matter when it dosnt
                if self.gl_encoding == GoldlabelEncodings.dense:
                    decode_func = dense2linepoints
                elif self.gl_encoding == GoldlabelEncodings.onehot:
                    decode_func = sparse2linepoints
                else:
                    print("TODO unsupported gl_encoding in dataset.show: ", self.gl_encoding)
                    return None
                points = decode_func(gl, x_size=(h, w))
                predPoints = []
                #print("Dataloader.Dataset.show: linepoint: (h, w) = ", (h, w))
                print("Dataloader.Dataset.show: linepoint: points = ", points)
                if predicted is not None:
                    predPoints = decode_func(predicted[i], x_size=(h, w))
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
                    cv2.circle(img, point[1], point[2]//2, 125, 2)
                    cv2.line(img, point[0], point[1], 125, thickness=2)
                cv2.imshow(str(gl), img)
                cv2.waitKey(0)
            elif self.gl_type == GoldlabelTypes.text:
                if self.gl_encoding == GoldlabelEncodings.dense:
                    decode_func = dense2txt
                elif self.gl_encoding == GoldlabelEncodings.onehot:
                    decode_func = sparse2txt
                else:
                    print("TODO unsupported gl_encoding in dataset.show: ", self.gl_encoding)
                    return None
                gl = decode_func(gl)
                if predicted is not None:
                    print("Dataloader.Dataset: pred = ", getType(predicted[i]))
                    print("pred = ", decode_func(predicted[i]))
                print("gl = ", gl)
                cv2.imshow(str(gl), img)
                cv2.waitKey(0)
            elif self.gl_type == GoldlabelTypes.lineimg:
                cv2.imshow("img", img)
                for igl in range(len(gl)):
                    cv2.imshow("lineimg_"+str(igl), np.array(gl[igl], dtype="uint8"))
                cv2.waitKey(0)


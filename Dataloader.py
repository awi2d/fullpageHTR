import cv2
import numpy as np
import tensorflow as tf
import random
# import tensorflow_datasets as tf_ds
# TODO https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# TODO tf.keras.utils.image_dataset_from_directory nutzen
# 125 GB RAM auf server (gesamt)

line_point = ((int, int), (int, int), int)  # (startpoint of line, endpoint of line, height)

# data_dir = "../SimpleHTR/data/trainingDataset/"
data_dir = "C:/Users/Idefix/PycharmProjects/SimpleHTR/data/"  # The dirctory that is mapped to not be in the docker
iam_dir = data_dir + "iam/"  # the unchanged iam dataset
dataset_dir = data_dir + "generated/"  # directoy for storing generated data
models_dir = data_dir + "models/"  # directoy for storing trained models


class GoldlabelTypes:
    text = 0
    linepositions = 1
    number_of_lines = 2


class GoldlabelEncodings:
    onehot = 0
    dense = 1


class ImgTypes:
    word = 0
    line = 1
    paragraph = 2


class DatasetNames:
    iam = 0


def downscale(img, goldlabel: [line_point], x: int, y: int, gl_type=GoldlabelTypes.linepositions):
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
    if gl_type==GoldlabelTypes.linepositions:
        goldlabel = [((int(x1/x), int(y1/y)), (int(x2/x), int(y2/y)), int(h/y)) for ((x1, y1), (x2, y2), h) in goldlabel]
    return np.array(img, dtype="uint8"), goldlabel


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
    name = type(x).__name__
    if name == 'list':
        return '['+str(len(x))+":"+getType(x[0])+']'  # assumes all element of the list have the same type
    if name == 'tuple':
        r = '<'+str(len(x))+":"
        for i in x:
            r += getType(i)+'; '
        return r[:-2] + '>'
    if name == 'ndarray':
        return 'ndarray('+str(x.shape)+': '+(getType(x[0]) if len(x) > 0 else "Nix")+')'
    if name == 'BatchDataset':
        return str(name)+" : "+str(len(x))
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
        gl = gl
    elif gl_type == GoldlabelTypes.number_of_lines:
        gl = gl
    else:
        print("Dataloader.rotate_img: TODO unsupported goldlabel_type in Dataloader.rotate_img: ", gl_type)
        gl = "TODO"
    # rescale image to have original size
    scale = min(nW/w, nH/h)
    image, gl = downscale(image, gl, x=scale, y=scale, gl_type=gl_type)
    return image, gl


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


alphabet = np.array([' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
def txt2sparse(txt, y_size):
    #print("Dataloader.txt2sparse: txt ", len(txt), ", ysize ", y_size)
    assert len(txt) <= y_size
    txt = txt.lower()
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


def sample_linepoint(img, goldlabel: [line_point], upperleftcorner: (int, int), sampesize: (int, int)):
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
    print("Dataloader.sample_linepoint")
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

def fitsize(img, gl, w, h, gl_type=GoldlabelTypes.linepositions):
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
    print("Dataloader.fitsiz: (h, w) = ", (h, w))
    if gl_type == GoldlabelTypes.linepositions:
        if img.shape[0] >= h:
            img, gl = sample_linepoint(img, gl, (0, 0), (img.shape[0], h-1))
        if img.shape[1] >= w:
            img, gl = sample_linepoint(img, gl, (0, 0), (w-1, img.shape[1]))
    elif gl_type == GoldlabelTypes.text:
        img = np.array(img[0:h, 0:w], dtype="uint8")  # TODO text that is cutted out of the image should be cut out of the goldlabel
    else:
        return "TODO"
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
    that means they either get padded are cliped.
    :param y_size:
    size all goldlabel_encodings in data will have
    :return:
    [(image, goldlabel)] like input, but the goldlabel is encoded and image and goldlabel are padded to size
    """
    h = int(max([img.shape[0] for (img, gl) in data])/2)  # konstants should always be the same as in rescaling. This will obviusly break with goldlabel_type == text
    w = int(max([img.shape[1] for (img, gl) in data])/4)
    print("Dataloader.encode_and_pad: natural w, h = ", h, ", ", w)
    if size is not None:
        h = size[0]
        w = size[1]
    print("Dataloader.encode_and_pad: used w, h = ", h, ", ", w)
    data = [downscale(img, points, 4, 2, gl_type=goldlabel_type) for (img, points) in data]  # TODO should be optional
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
        data = [fitsize(img, gl, w, h) for (img, gl) in data]
        maxlinecount = max([gl for (img, gl) in data])
        if goldlabel_encoding == GoldlabelEncodings.dense:
            return [(np.pad(img, ((0, h-img.shape[0]), (0, w-img.shape[1])), mode='constant', constant_values=255), lc/maxlinecount) for (img, lc) in data]
        elif goldlabel_encoding == GoldlabelEncodings.onehot:
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
        if goldlabel_type == GoldlabelTypes.text:
            goldlabel = '<br>'.join(goldlabel_list)
        elif goldlabel_type == GoldlabelTypes.linepositions:
            goldlabel = [((0, 0), (0, 0), 0)]*len(goldlabel_list)
            offset = 0
            for i_gl in range(len(goldlabel_list)):  # gl is list of points
                #print("goldlabe_list[", i_gl, "] = ", str(goldlabel_list[i_gl]))
                pre_gl = goldlabel_list[i_gl][0]  # (startpoint, endpoint, height)
                goldlabel[i_gl] = ((pre_gl[0][0], pre_gl[0][1]+offset), (pre_gl[1][0], pre_gl[1][1]+offset), pre_gl[2])
                #offset += pad[i_gl]+pre_gl[2]
                offset += img_list[i_gl].shape[0]
        else:
            print("Dataloader.concat_data: goldlabel_type ", goldlabel_type, " is not valid")
    elif axis == 1:
        h = max([img.shape[0] for img in img_list])
        img_list = [np.pad(img_list[i], ((int(np.floor_divide((h-img_list[i].shape[0]),2)), int(np.ceil((h-img_list[i].shape[0])/2))), (pad[i], 0)), mode='constant', constant_values=255) for i in range(len(img_list))]  # white padding added at bottom
        if goldlabel_type == GoldlabelTypes.text:
            goldlabel = ' '.join(goldlabel_list)
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



def getTrainingData(goldlabel_encoding=GoldlabelEncodings.onehot):
    """
    :param goldlabel_encoding:
    from Dataset.goldlabel_encodings
    :return:
    (x_train, y_train), (x_val, y_val), (x_test), (y_test), so that tf.model.fit(x_train, y_train, validation_data=(x_val, y_val)) works.
    """
    data = getData(dir=data_dir, dataset_loader=DatasetNames.iam, img_type=ImgTypes.paragraph, goldlabel_type=GoldlabelTypes.linepositions, goldlabel_encoding=goldlabel_encoding, maxcount=100, x_size=(512, 1024))
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
    words_per_line = [2, 3]
    lines_per_paragrph = [3, 4]

    if dataset_loader == DatasetNames.iam:
        data = load_iam(dir, goldlabel_type)  # [(relative path of img file, goldlabel text of that file)]
    else:
        raise "invalid dataset_loader: "+str(dataset_loader)
    #print("path_gl: ", getType(data))
    #print("data_imgpath_goldlabel = ", data[:5])
    # if goldlabel_type = text: type(data) = [(img: np.array(?, ?), text: string)]
    # if goldlabel_type = linepositions: type(data) = [(img: np.array(?, ?), [point: (int, int)])]

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
    data = [(load_img(dir+"/"+path), gl) for (path, gl) in data]
    #print("imgword_gl: ", getType(data))
    if img_type == ImgTypes.word:
        if goldlabel_type == GoldlabelTypes.linepositions:
            ys = 1
        elif goldlabel_type == GoldlabelTypes.text:
            ys = 32  # max([len(d[1]) for d in data])
        else:
            ys = 1
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=ys)
        print("imgwordenc_gl: ", getType(data))
        return data

    tmp = [data[i:i+random.choice(words_per_line)] for i in range(0, len(data), max(words_per_line))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    word_distance = [10, 20]
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=1, pad=[random.choice(word_distance) for unused in range(len(t))]) for t in tmp]
    data = [rotate_img(img, lp, goldlabel_type, angle=random.randrange(-line_para_winkel[0], line_para_winkel[0]+1, 1)) for (img, lp) in data]
    #print("data_lines[0]: ", data[0])
    #print("data_imgline_goldlabel = ", data[:5])
    #print("imgline_gl: ", getType(data))
    if img_type == ImgTypes.line:  # line
        if goldlabel_type == GoldlabelTypes.linepositions:
            ys = 1
        elif goldlabel_type == GoldlabelTypes.text:
            ys = 32  # max([len(d[1]) for d in data])
        else:
            ys = 1
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=ys)
        print("imglineenc_gl: ", getType(data))
        return data
    # rotate lines

    tmp = [data[i:i+random.choice(lines_per_paragrph)] for i in range(0, len(data), max(lines_per_paragrph))]  # tmp[0] = (list of words_per_line pictures, list of their goldlabels)
    line_distance = [5, 10]
    data = [concat_data(t, goldlabel_type=goldlabel_type, axis=0, pad=[random.choice(line_distance) for unused in range(len(t))]) for t in tmp]
    data = [rotate_img(img, lp, goldlabel_type, angle=random.randrange(-line_para_winkel[1], line_para_winkel[1]+1, 1)) for (img, lp) in data]
    #print("imgpara_gl: ", getType(data))
    #print("data_parag[0]: ", data[0])
    #print("data_imgpara_goldlabel = ", data[:5])
    if img_type == ImgTypes.paragraph:  # paragraph
        if goldlabel_type == GoldlabelTypes.linepositions:
            ys = max(lines_per_paragrph)
        elif goldlabel_type == GoldlabelTypes.text:
            ys = max([len(d[1]) for d in data])
        else:
            ys = 1
        data = encode_and_pad(data, goldlabel_type, goldlabel_encoding, size=x_size, y_size=ys)
        #print("imgparaenc_gl: ", getType(data))
        return data
    return "Dataloader.getData: This return statement is impossible to reach."

class Dataset_test:
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

class RNNDataset:
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

class Dataset:
    name = "real"
    data_directory = None
    gl_type = 0
    gl_encoding = 0
    pos = 0
    dataset_size = -1
    imgsize = (32, 32)
    add_empty_images = True

    def __init__(self, datadir=data_dir, gl_encoding=GoldlabelEncodings.dense, gl_type=GoldlabelTypes.linepositions, img_type=ImgTypes.paragraph):
        self.data_directory = datadir
        self.gl_type = gl_type
        self.img_type = img_type
        self.gl_encoding = gl_encoding
        self.pos = 0
        self.dataset_size = len(load_iam(datadir, gl_type))
        if self.img_type == ImgTypes.line:
            self.imgsize = (32, 128)
            self.line_para_winkel = (10, 0)
        elif self.img_type == ImgTypes.paragraph:
            self.imgsize = (256, 256)
            self.line_para_winkel = (5, 10)
        else:
            self.imgsize = (32, 64)  # word or invalid input
            self.line_para_winkel = (0, 0)
        self.imgsize = None

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
        x_train = np.array([d[0] for d in data], dtype=float)
        y_train = np.array([d[1] for d in data], dtype=float)
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
                print("gl = ", gl)
            cv2.imshow(str(gl), img)
            cv2.waitKey(0)


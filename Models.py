import keras.losses
import numpy as np
import tensorflow as tf

import Dataloader


def fullyconnectedFedforward(in_shape=(1000, 2000), out_length=5, activation="linear"):
    #trainable params: (out_length = ?) 24,978, (out_length = 4*5) 5,247,020
    # i.g. in_shape[0]*in_shape[1]*out_length*2+out_length*out_length*8
    model = tf.keras.models.Sequential(name="fullyconnectedFedforward")
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # grayscale image has values in list(range(255))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=out_length*3, activation='relu'))
    model.add(tf.keras.layers.Dense(units=out_length*2, activation='relu'))
    model.add(tf.keras.layers.Dense(units=out_length, activation=activation))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.5)  # learning rate should be unused
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


def testRNN(in_shape=(32, 32), out_length=2, activation="linear"):
    model = keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    timestamps = None
    x_size = 3
    model.add(tf.keras.layers.Rescaling(1., input_shape=(timestamps, x_size)))

    # The output of LSTM will be a 3D tensor of shape (batch_size, timesteps, 3)
    model.add(tf.keras.layers.LSTM(3, return_sequences=True, activation="relu"))  # with return_sequence=False it would be (batch_size, 3)

    #model.add(tf.keras.layers.Dense(3))

    opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.5)  # learning rate should be unused
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


def fullyconnectedFedforward2(in_shape=(1000, 2000), out_length=6, activation='linear'):
    # should be exactly the same as fullyconnectedFedforward
    inputs = keras.Input(shape=in_shape)
    rescaledIn = tf.keras.layers.Rescaling(1./255)(inputs)
    flat = tf.keras.layers.Flatten()(rescaledIn)
    x1 = tf.keras.layers.Dense(units=out_length*3, activation='relu')(flat)
    x2 = tf.keras.layers.Dense(units=out_length*2, activation='relu')(x1)
    x3 = tf.keras.layers.Dense(units=out_length, activation=activation)(x2)
    outputs = tf.keras.layers.Dense(out_length, activation=activation, name="predictions")(x3)

    model = keras.Model(inputs=inputs, outputs=outputs, name="fullyconnectedFedforward_functional")
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


def cvff(in_shape=(1000, 2000), out_length=5, activation="linear"):
    #trainable params: ?
    activation_in = "relu"
    model = tf.keras.models.Sequential(name="cvff")
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # grayscale image has values in list(range(255))
    model.add(tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(3, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=out_length*3, activation='relu'))
    model.add(tf.keras.layers.Dense(units=out_length*2, activation='relu'))
    model.add(tf.keras.layers.Dense(units=out_length, activation=activation))  # mit tanh aks activation in der letzen schicht funktioniert das nicht
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.5)  # learning rate should be unused
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


def vgg11(in_shape, out_length, activation="linear"):
    # structure nearly copyed from paper VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION from Karen Simonyan & Andrew Zisserman
    #Trainable params: 1,256,222
    activation_in = "relu"
    model = tf.keras.models.Sequential(name="vgg11")
    model.add(tf.keras.layers.Rescaling(1./127.5, offset=-1, input_shape=in_shape))  # grayscale image has values in [0, 255] rescale to [-1, 1]
    #model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # grayscale image has values in [0, 255] rescale to [0, 1]
    dim = 16
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    dim *= 2
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    dim *= 2
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.LeakyReLU(alpha=0))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    dim *= 2
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.LeakyReLU(alpha=0))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), padding="same", activation=activation_in))
    model.add(tf.keras.layers.LeakyReLU(alpha=0))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(units=4*out_length, activation=activation_in))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=2*out_length, activation=activation_in))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=2*out_length, activation=activation_in))
    model.add(tf.keras.layers.Dense(units=out_length, activation=activation))  # should be softmax if goldlabel_encoding is onehot

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


# neue modelle:
# erst conv und averagePooling, dann dense
#
# filter anzahl/größer machen
def conv(in_shape=(32, 32), out_length=6, activation='linear'):
    # trainable parameters 23,316
    model = tf.keras.models.Sequential(name="conv")

    # Layer 1 Conv2D
    # model.add(Dropout(0.2, input_shape=input_shape))
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # rescale img to [0, 1]
    model.add(tf.keras.layers.Reshape((in_shape[0], in_shape[1], 1)))
    #model.add(tf.keras.layers.Rescaling(1./127.5, offset=-1, input_shape=in_shape))  # rescale img to [-1, 1]
    model.add(tf.keras.layers.Conv2D(24, (5, 5), strides=(1, 1), padding="same", activation='tanh', input_shape=in_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.4))

    # Layer 2 Pooling Layer
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Dropout(0.4))

    # Layer 3 Conv2D
    model.add(tf.keras.layers.Conv2D(12, (5, 5), strides=(1, 1), padding="same", activation='tanh'))  #
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.4))
    # model.add(BatchNormalization())

    # Layer 3 Pooling Layer
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(tf.keras.layers.Dropout(0.4))

    # Layer 4 Conv2D
    model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), padding="same", activation='tanh'))

    # Layer 5 Pooling Layer
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=120, activation='tanh'))
    model.add(tf.keras.layers.Dense(units=84, activation='tanh'))
    model.add(tf.keras.layers.Dense(units=out_length, activation=activation))

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


def simpleHTR(in_shape=(32, 256), out_length=len(Dataloader.alphabet), activation='relu'):
    """
    :param in_shape:
    :param out_length:
    :param activation:
    :return:
    model with shapes in_shape -> (in_shape[1]//2, out_length)
    """
    # TODO neues Modell: wie simpleHTR, aber bild und ausgabe vom linefinder als eingabe
    assert in_shape[0] == 32
    # should be the same as simpleHTR, but migrated to tensorflow2
    # https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5


    model = tf.keras.models.Sequential(name="simpleHTR")
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # rescale img to [0, 1]
    model.add(tf.keras.layers.Reshape((in_shape[0], in_shape[1], 1)))
    #model.add(tf.expand_dims(input=in_shape, axis=3))

    # setup_cnn
    model.add(tf.keras.layers.Conv2D(kernel_size=(5, 5), filters=32, padding='SAME', strides=(1, 1), activation=activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(5, 5), filters=64, padding='SAME', strides=(1, 1), activation=activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='SAME', strides=(1, 1), activation=activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='SAME', strides=(1, 1), activation=activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='SAME', strides=(1, 1), activation=activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='VALID'))
    #current tensor size: (None, 32, 1, 256)

    """setup_rnn"""
    model.add(tf.keras.layers.Reshape((in_shape[1]//4, 256)))
    #current tensor size: (None, 32, 256)
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))  # maybe not exactly the same as original simpleHTR
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
    #current tensor size: (None, 32, 512)
    # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
    model.add(tf.keras.layers.Reshape((in_shape[1]//4, 1, 512)))
    #current tensor size: (None, 1, 32, 1, 512)

    # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
    model.add(tf.keras.layers.Conv2D(kernel_size=(1, 1), filters=out_length,  padding='SAME', strides=(1, 1), dilation_rate=(2, 2)))
    #current tensor size: (None, 32, 1, 10)
    model.add(tf.keras.layers.Reshape((in_shape[1]//4, out_length)))
    #current tensor size: (None, 32, 10), should be (batch_size, timestemps==maxlength_of_output_text, sparse char encoding)


    """setup_ctc
    self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
    # ground truth text as sparse tensor
    self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]), tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))

    # calc loss for batch
    self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
    self.loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.ctc_in_3d_tbc,sequence_length=self.seq_len, ctc_merge_repeated=True))

    # calc loss for each element to compute label probability
    self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, len(self.char_list) + 1])
    self.loss_per_element = tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input, sequence_length=self.seq_len, ctc_merge_repeated=True)

    # best path decoding or beam search decoding

    self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)
    """

    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


def simpleHTR2(in_shape=(2048, 128, 1), out_length=len(Dataloader.alphabet), activation='relu', lr=3e-4):  # TODO add +1 to len(alphabet) ctc-blank label
    """
    :param in_shape: The size of the input to the network.
    :param out_length: The size of the output.
    maximum 128
    :param lr: The initial learning rate.
    :return: a tf.keras.model with almost the simpleHTR architekture
    """
    # copied from https://github.com/UniDuEChristianGold/LineRecognition
    input_data = tf.keras.layers.Input(name="input", shape=in_shape)
    cnn = tf.keras.layers.Rescaling(1./255, input_shape=in_shape)(input_data)  # rescale img to [0, 1]
    # ====================== Conv 0 ======================

    cnn = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 9), strides=(1, 1), padding="same")(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    # ====================== Conv 1 ======================

    cnn = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    # ====================== Conv 2 ======================

    cnn = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    # ====================== Conv 3 ======================

    cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(cnn)

    # ====================== Conv 4 ======================

    cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)
    cnn = tf.keras.layers.Conv2D(filters=80, kernel_size=(5, 5), strides=(1, 1), padding="same")(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # ====================== Conv 5 ======================

    cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)
    cnn = tf.keras.layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding="same")(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # ====================== Conv 6 ======================

    cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)
    cnn = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same")(cnn)

    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
    cnn = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    # Shape: (batch, new_rows, new_cols, filters)
    shape = cnn.get_shape()
    blstm = tf.keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    # ====================== BLSTM 0 ======================

    blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(blstm)
    blstm = tf.keras.layers.Activation('relu')(blstm)

    # ====================== BLSTM 1 ======================

    blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True))(blstm)
    blstm = tf.keras.layers.Activation('relu')(blstm)
    blstm = tf.keras.layers.Dropout(rate=0.5)(blstm)

    # ====================== Dense 0 ======================

    output_data = tf.keras.layers.Dense(units=out_length, activation="softmax")(blstm)

    model = keras.Model(inputs=input_data, outputs=output_data, name="simpleHTR2")
    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


class linepoint2lineimg(tf.keras.layers.Layer):
    # https://www.tensorflow.org/text/tutorials/nmt_with_attention
    # img, linepoint -> line of text that is shown in that image
    # used to kit linefinder models (conv, cvff, cgg11) and linereader models (simpleHTR, simpleHTR2) together.
    name = "linepoint2lineimg"

    def __init__(self, img_shape):
        super().__init__(dynamic=True, trainable=False)
        print("Models.linepoint2lineimg: init(", img_shape, ")")
        self.imgshape = img_shape

    def build(self, batch_input_shape):
        print("Models.linepoint2lineimg: build(", batch_input_shape, ")")

    def compute_output_shape(self, unknwon):
        print("Models.linepoint2lineimg.compute_output_shape: unknwon = ", unknwon)
        numlines = 7
        return (None, numlines, 32, 256)

    def call(self, linepoints, image):
        print("Models.linepoint2lineimg: call(", Dataloader.getType(linepoints), ", ", Dataloader.getType(image), ")")
        #linepoints = Dataloader.dense2linepoints(linepoints, max_x=self.imgshape[0], max_y=self.imgshape[1])
        #assert linepoints.shape[1]%5 == 0
        #max_x = self.imgshape[0]
        #max_y = self.imgshape[1]
        #linepoints = [((int(linepoints[i]*max_x), int(linepoints[i+1]*max_y)), (int(linepoints[i+2]*max_x), int(linepoints[i+3]*max_y)), int(linepoints[i+4]*max_y)) for i in range(0, linepoints.shape[1], 5)]
        #print("linepoints = ", linepoints)
        #linepoints = [((int(linepoints[i]*max_x), int(linepoints[i+1]*max_y)), (int(linepoints[i+2]*max_x), int(linepoints[i+3]*max_y)), max(1, int(linepoints[i+4]*max_y))) for i in range(0, linepoints.shape[1], 5)]
        #TODO extracline is not supported:  OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed: AutoGraph did convert this function. This might indicate you are trying to use an unsupported feature.
        return [image[:][0:32][0:256] for i in range(7)]
        #return [Dataloader.extractline(image, linepoints[i:5*i], max_x=self.imgshape[0], max_y=self.imgshape[1]) for i in range(linepoints.shape[1]//5)]


def test_multioutput():
    # train on np.array((batch_size, input_size)) -> {name_of_output_layer: np.array((batch_size, output_size)), ...}
    input_data = tf.keras.layers.Input(name="input", shape=(2, 2))
    #output_data = [tf.keras.layers.Flatten()(input_data) for _ in range(1)]
    output_data = tf.keras.layers.Flatten()(input_data)
    output_data1 = tf.keras.layers.Dense(units=2, name="output_for_line_1")(output_data)
    output_data2 = tf.keras.layers.Dense(units=2, name="output_for_line_2")(output_data)

    model = keras.Model(inputs=input_data, outputs=[output_data1, output_data2], name="test_multioutput")
    opt = tf.keras.optimizers.Adam(learning_rate=2**-8, beta_1=0.5)
    loss = {"output_for_line_1": keras.losses.MeanSquaredError(), "output_for_line_2": keras.losses.MeanSquaredError()}
    model.compile(loss=loss, optimizer=opt)
    return model


def linefinder(in_shape=(256, 256), output_shape=(2, 32, 256), activation='relu'):
    """
    :param input_shape:
    shape of paragrph image. all values should be powers of 2
    :param output_shape:
    (number of lines, shape of lineimg[0], shape of lineimg[1])
    all values exept number of lines should be powers of 2
    :return:
    """
    # image of paragraph -> [image of line]
    input_data = tf.keras.layers.Input(name="input", shape=in_shape)
    cnn = tf.keras.layers.Reshape((in_shape[0], in_shape[1], 1))(input_data)
    current_shape = cnn.get_shape()
    while current_shape[1] != output_shape[1] or current_shape[2] != output_shape[2]:
        print("current shape = ", current_shape)
        poolsize = (1, 1)
        if current_shape[1] > output_shape[1]:
            poolsize = (2, 1)
        if current_shape[2] > output_shape[2]:
            poolsize = (poolsize[0], 2)
        cnn = tf.keras.layers.Dropout(rate=0.2)(cnn)
        cnn = tf.keras.layers.Conv2D(filters=output_shape[0], kernel_size=(5, 5), strides=(1, 1), padding="same")(cnn)

        cnn = tf.keras.layers.BatchNormalization()(cnn)
        cnn = tf.keras.layers.LeakyReLU(alpha=0.01)(cnn)
        cnn = tf.keras.layers.MaxPooling2D(pool_size=poolsize, strides=poolsize, padding="valid")(cnn)
        current_shape = cnn.get_shape()

    #flat = tf.keras.layers.Flatten()(cnn)
    #outputs = [tf.keras.layers.Dense(units=output_shape[1]*output_shape[2], activation=activation, name="line"+str(i))(flat) for i in range(output_shape[0])]

    reshaped = tf.keras.layers.Reshape((output_shape[0], output_shape[1], output_shape[2], 1))(cnn)
    outputs = [tf.keras.layers.Conv2D(filters=1, kernel_size=(9, 9), strides=(1, 1), padding="same", activation=activation, name="line"+str(i))(reshaped[:,i,:,:]) for i in range(output_shape[0])]

    losses = {}
    for i in range(output_shape[0]):
        losses["line"+str(i)] = keras.losses.MeanSquaredError()
    model = keras.Model(inputs=input_data, outputs=outputs, name="linefinder_model")
    opt = tf.keras.optimizers.Adam(learning_rate=2**-8, beta_1=0.5)
    model.compile(loss=losses, optimizer=opt)
    return model

class customlayer(tf.keras.layers.Layer):

    def __init__(self, name):
        print("Models.customlayer: name = ", name)
        super(customlayer, self).__init__(name=name)
        self.out_size = (32, 64)

    def build(self, batch_input_shape):
        print("Models.customlayer: build(", batch_input_shape, ")")

    def compute_output_shape(self, unknwon):
        return self.out_size

    def call(self, img, pos):
        """
        :param img:
        tf.tensor(shape=(x, y))
        the paragraph image
        :param pos:
        tf.tensor(shape(2))
        the upper left corner of the line
        :return:
        the slice of image that contions line
        """
        print("Models.customlayer: call img=", Dataloader.getType(img)+", pos=", Dataloader.getType(pos))
        bi = 0
        # pos[bi][0] has to be int, but casting it to int leads to "ValueError: No gradients provided"
        #pos = tf.cast(pos, dtype=tf.int32)
        r = np.zeros((32, 256))
        for i in range(len(r)):
            for j in range(len(r[i])):
                r[i][j] = 0
                for ii in range(img.shape[0]):
                    for ij in range(img.shape[1]):
                        weigth = 1/(abs(i+pos[0]-ii)**2+abs(j+pos[1]-ij)**2)  # weight = 1/distance of requested pixel with (ii, ij)
                        r[i][j] += weigth*img[ii][ij]
        #return tf.slice(img, begin=(0, pos[bi][0], pos[bi][1]), size=(-1, self.out_size[0], self.out_size[1]))
        return r


def total(in_shape, linefindermodel, linereadermodel, out_shape=len(Dataloader.alphabet)):
    input_data = tf.keras.layers.Input(name="input", shape=in_shape)
    lineimgs = linefindermodel(input_data)
    texts = [linereadermodel(lineimg) for lineimg in lineimgs]
    output = tf.keras.layers.Concatenate(axis=1)(texts)
    model = keras.Model(inputs=input_data, outputs=output, name="total")
    opt = tf.keras.optimizers.Adam(learning_rate=2**-8, beta_1=0.5)  # what even is beta_1 of Adam?
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=opt)
    return model

# maybe use model_conv[-2].outputs as inputs to attention
def oldtotal(in_shape=(256, 256), out_length=6, activation='linear'):
    input_data = tf.keras.layers.Input(name="input", shape=in_shape)
    linepoints = conv(in_shape=in_shape, out_length=out_length, activation=activation)(input_data)
    lineimgs = customlayer(name="mycustomlayer")(input_data, linepoints)
    #lineimgs = linepoint2lineimg(img_shape=(32, 256))(linepoints, input_data)
    print("Models.total: lineimgs = ", Dataloader.getType(lineimgs))
    # lineimgs = [32x256 image of first line, 32x256 image of second line, ...]
    #lineimgs = [tf.keras.layers.Attention()([linepoints[i*5:(i+1)*5], input_data]) for i in range(out_length)]  # outputshape is the same as linepoints[i*5:(i+1)*5]
    #simplehtrmode_as_layer = simpleHTR(in_shape=(32, 256))
    #lines = [simplehtrmode_as_layer(lineimg) for lineimg in lineimgs]
    model = keras.Model(inputs=input_data, outputs=lineimgs, name="simpleHTR2")
    opt = tf.keras.optimizers.Adam(learning_rate=2**-8, beta_1=0.5)
    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=opt)
    return model

# https://www.tensorflow.org/tutorials/generative/pix2pix
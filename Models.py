import keras.losses
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


def simpleHTR(in_shape=(32, 128), out_length=len(Dataloader.alphabet), activation='relu'):
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
        super().__init__()
        print("Models.linepoint2lineimg: init(", img_shape, ")")
        self.imgshape = img_shape

    def build(self, input_shape):
        print("Models.linepoint2lineimg: build(", input_shape, ")")

    def call(self, linepoints, image):
        print("Models.linepoint2lineimg: call(", linepoints, ", ", image, ")")
        linepoints = Dataloader.dense2linepoints(linepoints, max_x=self.imgshape[0], max_y=self.imgshape[1])
        return [Dataloader.extractline(image, lp, max_x=self.imgshape[0], max_y=self.imgshape[1]) for lp in linepoints]


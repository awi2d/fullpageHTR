import keras.losses
import tensorflow as tf


def fullyconnectedFedforward(in_shape=(1000, 2000), out_length=5, activation="linear"):
    #trainable params: (out_length = ?) 24,978, (out_length = 4*5) 5,247,020
    # i.g. in_shape[0]*in_shape[1]*out_length*2+out_length*out_length*8
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # grayscale image has values in list(range(255))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=out_length*3, activation='relu'))
    model.add(tf.keras.layers.Dense(units=out_length*2, activation='relu'))
    model.add(tf.keras.layers.Dense(units=out_length, activation=activation))  # mit tanh aks activation in der letzen schicht funktioniert das nicht
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

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt)  # metrics=['mean_squared_error']
    return model


def cvff(in_shape=(1000, 2000), out_length=5, activation="linear"):
    #trainable params: ?
    activation_in = "relu"
    model = tf.keras.models.Sequential()
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
    model = tf.keras.models.Sequential()
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
def conv(in_shape=(32, 32, 1), out_length=6, activation='linear'):
    # trainable parameters 23,316
    model = tf.keras.models.Sequential()

    # Layer 1 Conv2D
    # model.add(Dropout(0.2, input_shape=input_shape))
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # rescale img to [0, 1]
    #model.add(tf.keras.layers.Rescaling(1./127.5, offset=-1, input_shape=in_shape))  # rescale img to [-1, 1]
    model.add(tf.keras.layers.Conv2D(24, (5, 5), strides=(1, 1), padding="same", input_shape=in_shape, activation='tanh'))
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
    model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

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


def simpleHTR(in_shape=(32, 128, 1), out_length=27, activation='relu'):
    # Eingabe auf ganzen Bild+Linepoint ändern.
    assert in_shape[0] == 32
    # should be the same as simpleHTR, but migrated to tensorflow2
    # https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # rescale img to [0, 1]
    #model.add(tf.expand_dims(input=in_shape, axis=3))

    # setup_cnn
    model.add(tf.keras.layers.Conv2D(kernel_size=(5, 5), filters=32, padding='SAME', strides=(1, 1), activation=activation))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(5, 5), filters=64, padding='SAME', strides=(1, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='SAME', strides=(1, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='SAME', strides=(1, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='VALID'))

    model.add(tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='SAME', strides=(1, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='VALID'))
    #current tensor size: (None, 32, 1, 256)

    """setup_rnn"""
    model.add(tf.keras.layers.Reshape((32, 256)))
    #current tensor shape (None, 32, 256)
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))  # maybe not exactly the same as original simpleHTR
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
    #current tensor size: (None, 32, 512)
    # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
    model.add(tf.keras.layers.Reshape((32, 1, 512)))
    #current tensor size: (None, 1, 32, 1, 512)

    # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
    model.add(tf.keras.layers.Conv2D(kernel_size=(1, 1), filters=out_length,  padding='SAME', strides=(1, 1), dilation_rate=(2, 2)))
    #current tensor size: (None, 32, 1, 10)
    model.add(tf.keras.layers.Reshape((32, out_length)))
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


# copied from EasyHTR
import numpy as np
import os
Batch = str  # TODO

class SimpleHTR:
    """Minimalistic TF model for HTR."""

    def __init__(self,
                 char_list: [str],
                 must_restore: bool = False,
                 dump: bool = False, model_dir = '../model/word') -> None:

        """Init model: add CNN, RNN and CTC and initialize TF."""
        self.dump = dump
        self.char_list = char_list
        self.must_restore = must_restore
        self.snap_ID = 0
        self.model_dir = model_dir

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        # setup CNN, RNN and CTC
        self.setup_cnn()
        self.setup_rnn()
        self.setup_ctc()

        # setup optimizer to train NN
        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        # initialize TF
        self.sess, self.saver = self.setup_tf()

    def setup_cnn(self) -> None:
        """Create CNN layers."""
        cnn_in4d = tf.expand_dims(input=self.input_imgs, axis=3)

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # create layers
        pool = cnn_in4d  # input to first CNN layer
        for i in range(num_layers):
            kernel = tf.Variable(tf.random.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]], stddev=0.1))
            print("easyHTR: kernal = ", kernel)
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1), strides=(1, stride_vals[i][0], stride_vals[i][1], 1), padding='VALID')
        self.cnn_out_4d = pool

    def setup_rnn(self) -> None:
        """Create RNN layers."""
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in3d,
                                                                dtype=rnn_in3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.random.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_out_3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])

    def setup_ctc(self) -> None:
        """Create CTC loss and decoder."""
        # BxTxC -> TxBxC
        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.ctc_in_3d_tbc,
                                                  sequence_length=self.seq_len,
                                                  ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32,
                                                        shape=[None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input,
                                                         sequence_length=self.seq_len, ctc_merge_repeated=True)

        # best path decoding or beam search decoding

        self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)

    def setup_tf(self) -> (tf.compat.v1.Session, tf.compat.v1.train.Saver):
        """Initialize TF."""
        #print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        latest_snapshot = tf.train.latest_checkpoint(self.model_dir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + self.model_dir)

        # load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

    def to_sparse(self, texts: [str]) -> ([[int]], [int], [int]):
        """Put ground truth texts into sparse tensor for ctc_loss."""
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for batchElement, text in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for i, label in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoder_output_to_text(self, ctc_output: tuple, batch_size: int) -> [str]:
        """Extract texts from output of CTC decoder."""

        # word beam search: already contains label strings

        # TF decoders: label strings are contained in sparse tensor
        # ctc returns tuple, first element is SparseTensor
        decoded = ctc_output[0][0]
        # contains string of labels for each batch element
        label_strs = [[] for _ in range(batch_size)]
        # go over all indices and save mapping: batch -> values
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batch_element = idx2d[0]  # index according to [b,t]
            label_strs[batch_element].append(label)

        # map labels to chars for all batch elements
        return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]

    def train_batch(self, batch: Batch) -> float:
        """Feed a batch into the NN to train it."""
        num_batch_elements = len(batch.imgs)
        max_text_len = batch.imgs[0].shape[0] // 4
        sparse = self.to_sparse(batch.gt_texts)
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_imgs: batch.imgs, self.gt_texts: sparse,
                     self.seq_len: [max_text_len] * num_batch_elements, self.is_train: True}
        _, loss_val = self.sess.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_val

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        """Dump the output of the NN to CSV file(s)."""
        dump_dir = '../dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        # iterate over all batch elements and create a CSV file for each one
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                for c in range(max_c):
                    csv += str(rnn_output[t, b, c]) + ';'
                csv += '\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

    def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
        """Feed a batch into the NN to recognize the texts."""

        # decode, optionally save RNN output
        num_batch_elements = len(batch.imgs)

        # put tensors to be evaluated into list
        eval_list = []

        eval_list.append(self.decoder)

        if self.dump or calc_probability:
            eval_list.append(self.ctc_in_3d_tbc)

        # sequence length depends on input image size (model downsizes width by 4)
        max_text_len = batch.imgs[0].shape[0] // 4

        # dict containing all tensor fed into the model
        feed_dict = {self.input_imgs: batch.imgs, self.seq_len: [max_text_len] * num_batch_elements,
                     self.is_train: False}

        # evaluate model
        eval_res = self.sess.run(eval_list, feed_dict)

        # TF decoders: decoding already done in TF graph

        decoded = eval_res[0]
        # word beam search decoder: decoding is done in C++ function compute()

        # map labels (numbers) to character string
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gt_texts) if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
                         self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}
            loss_vals = self.sess.run(eval_list, feed_dict)
            probs = np.exp(-loss_vals)

        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dump_nn_output(eval_res[1])

        return texts, probs

    def save(self) -> None:
        """Save model to file."""
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', global_step=self.snap_ID)

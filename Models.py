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

def fullyconnectedFedforward2(in_shape=(32, 32, 1), out_length=6, activation='linear'):
    # should be exactly the same as fullyconnectedFedforward
    inputs = keras.Input(shape=in_shape, name="digits")
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

#inputs = keras.Input(shape=(784,), name="digits")
#x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
#x = layers.Dense(64, activation="relu", name="dense_2")(x)
#outputs = layers.Dense(10, activation="softmax", name="predictions")(x)


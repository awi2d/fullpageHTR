import keras.losses
import tensorflow as tf


def getModel(name, input_size, output_length, loss=keras.losses.MeanSquaredError()):
    """
    :param name:
    Find-Follow-Read-lite:
    simpleHTR:
    :return:
    A model with the defined architekture, that takes in an image and returns text
    """
    if name not in name_modelfunc.keys():
        print("Models.getModel: name ", name, " is not in ", name_modelfunc.keys())
        return None
    return name_modelfunc[name](in_shape=input_size, out_length=output_length)

def findfollowreadlite_dense(in_shape=(1000, 2000), out_length=5):
    """
    :param in_shape:
    (int, int): shape of the input image. This model will only work on grayscale images that have exacly this size
    :param out_length:
    the number of output neuros. currently should be len(encoding_one_linepoint)*maximum_number_of_lines
    :return:
    a tenserflow modell
    """
    inputs = keras.Input(shape=(None, in_shape[0], in_shape[1],), name="digits")  # keine ahnung warum (1,2,) statt (1,2)
    in_rescaled = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(in_rescaled)
    outputs = tf.keras.layers.Dense(out_length, activation="tanh")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
    loss_func = keras.losses.MeanSquaredError()
    model.compile(loss=loss_func, optimizer=opt)
    return model


# TODO learning rate anpassen
# eingabe kleiner machen.
# filter anzahl/größer machen
def findfollowreadlite(in_shape=(32, 128, 1), out_length=26, lr=0.0001, lossfunc=keras.losses.MeanSquaredError()):
    model = tf.keras.models.Sequential()

    # Layer 1 Conv2D
    # model.add(Dropout(0.2, input_shape=input_shape))
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=in_shape))  # grayscale image has values in list(range(255))
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
    model.add(tf.keras.layers.Dense(units=out_length, activation='sigmoid'))

    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    model.compile(loss=lossfunc, optimizer=opt)  # metrics=['mean_squared_error']
    return model


def findfollowreadlite_cce(in_shape=(1000, 2000), out_length=5):
    return findfollowreadlite(in_shape=in_shape, out_length=out_length, lossfunc=keras.losses.CategoricalCrossentropy())


def findfollowreadlite_mse(in_shape=(1000, 2000), out_length=5):
    return findfollowreadlite(in_shape=in_shape, out_length=out_length, lossfunc=keras.losses.MeanSquaredError())


name_modelfunc = {"findfollowreadlite_cce": findfollowreadlite_cce, "findfollowreadlite_mse": findfollowreadlite_mse, "findfollowreadlite_dense": findfollowreadlite_dense}

#TODO learn "Functional API" https://www.tensorflow.org/guide/keras/train_and_evaluate
#inputs = keras.Input(shape=(784,), name="digits")
#x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
#x = layers.Dense(64, activation="relu", name="dense_2")(x)
#outputs = layers.Dense(10, activation="softmax", name="predictions")(x)


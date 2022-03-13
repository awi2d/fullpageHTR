import numpy as np
import tensorflow as tf
from keras import backend
import time
import cv2

import Dataloader
import Models


# <debug functions>

def save_dict(history, name="unnamed_model"):
    # assert len(history["loss"]) == len(history["val_loss"]) == len(history["lr"])
    with open(Dataloader.models_dir + str(name) + ".txt", 'w') as f:
        for k in history.keys():
            for e in history[k]:
                print(e, file=f)
            print(k, file=f)


def read_dict(name):
    with open(Dataloader.models_dir + str(name) + ".txt", 'r') as f:
        r = {}
        tmp = []
        for line in f.readlines():
            line = line.replace("\n", "")
            try:
                f = float(line)
                tmp.append(f)
            except:
                r[line] = tmp
                tmp = []
    assert len(r["loss"]) == len(r["val_loss"]) == len(r["lr"])
    return r


def show_trainhistory(history, name="unnamed model"):
    """
    :param history:
    a history, aka a dict with keys ["loss", "val_loss", "lr"] and len(history["loss"]) == len(history["val_loss"]) == len(history["lr"])
    :param name:
    the name of the file and the picture
    :return:
    saves a picture of the history in the models_dir
    """
    assert len(history["loss"]) == len(history["val_loss"]) == len(history["lr"])
    import matplotlib.pyplot as ploter  # TODO import should be at start of file
    # copied from https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales, second answer
    # Create figure and subplot manually
    # fig = plt.figure()
    # host = fig.add_subplot(111)

    # More versatile wrapper
    fig, host = ploter.subplots(figsize=(15, 10))  # (width, height) in inches
    # (see https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.subplots.html)

    par1 = host.twinx()
    try:
        host.set_xlim(0, len(history["loss"]))
        par1.set_ylim(0, max(max(history["loss"][2:]), max(history["val_loss"][2:])))
        host.set_ylim(0, max(history["lr"]))
    except:
        print("cant show trainhistory of " + str(name))
        return None

    host.set_xlabel("Epochs")
    par1.set_ylabel("loss")
    host.set_ylabel("learning_rate")

    color1 = ploter.cm.viridis(0)
    color2 = ploter.cm.viridis(0.5)
    color3 = ploter.cm.viridis(.9)

    xtmp = list(range(len(history["loss"])))
    p1, = par1.plot(xtmp, history["loss"], color=color1, label="training loss")
    p2, = par1.plot(xtmp, history["val_loss"], color=color2, label="validation loss")
    p3, = host.plot(xtmp, history["lr"], color=color3, label="learning rate")

    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='best')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    fig.tight_layout()

    # Best for professional typesetting, e.g. LaTeX
    ploter.savefig(Dataloader.models_dir + name + ".pdf")
    # For raster graphics use the dpi argument. E.g. '[...].png", dpi=200)'


# </debug functions>


def train(model, saveName, dataset, val, start_lr=2**(-8)):
    start_time = time.time()
    # TODO find better batch_size
    batch_size = 10
    if val is None or len(val) == 0:
        val = dataset.get_batch(batch_size)
    # print("train: ", x_train[0], " -> ", y_train[0])
    print("x.shape = ", val[0][0].shape)
    print("y.shape = ", val[1][0].shape)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=30, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    lr = start_lr
    lr_mult = [0.5, 1,
               2]  # at every step: train with all [lr*m for m in lr_mult] learning rates, pick the weights and lr that has the best validation_loss
    print("learning rate: ", lr, end=' ')
    history = {"loss": [], "val_loss": [], "lr": []}
    old_lr = -1
    older_lr = -2
    epochs_without_improvment = 0
    valLoss = 1
    x_train, y_train = dataset.get_batch(batch_size)
    epochcount = 0
    long_epochs = 16
    short_epochs = 1
    best_model = (valLoss, model.get_weights())
    while True:

        if lr == old_lr or lr == older_lr:
            # train for long
            print(str(lr) + "L", end=' ')
            backend.set_value(model.optimizer.learning_rate, lr)
            history_next = model.fit(x_train, y_train, epochs=long_epochs, callbacks=[callback], validation_data=val, verbose=0).history
            del x_train
            del y_train
            old_lr = -1
            older_lr = -2
            x_train, y_train = dataset.get_batch(batch_size)
        else:
            # train with different learning rates.
            # print("test lrs = ", [lr*m for m in lr_mult])
            weights_pre = model.get_weights()
            weigths_post = [({}, [])] * len(lr_mult)
            print(str(lr) + "T", end=' ')
            for i in range(len(lr_mult)):
                backend.set_value(model.optimizer.learning_rate, lr * lr_mult[i])
                tmphistory = model.fit(x_train, y_train, epochs=short_epochs, validation_data=val, verbose=0)
                # print("history: ", history.history)
                weigths_post[i] = (tmphistory.history, model.get_weights())
                model.set_weights(weights_pre)
            # pick best learning rate.
            best = 0
            # print("testing_lr: ", [(lr*lr_mult[i], weigths_post[i][0]['val_loss'][-1]) for i in range(len(lr_mult))])
            for i in range(1, len(lr_mult)):
                if weigths_post[i][0]['val_loss'][-1] < weigths_post[best][0]['val_loss'][-1]:
                    best = i
            model.set_weights(weigths_post[best][1])
            history_next = weigths_post[best][0]
            del weigths_post
            older_lr = old_lr
            old_lr = lr
            lr = lr * lr_mult[best]
        epochcount += len(history_next['loss'])
        if history_next['val_loss'][-1] >= best_model[0]:  # in diesem schritt hat sich nichst verbesser
            epochs_without_improvment += len(history_next['loss'])

        history['val_loss'] = history['val_loss'] + history_next['val_loss']
        history['loss'] = history['loss'] + history_next['loss']
        history['lr'] = history['lr'] + [lr] * len(history_next['loss'])
        valLoss = history['val_loss'][-1]
        if valLoss < best_model[0]:
            epochs_without_improvment = 0
            best_model = (valLoss, model.get_weights())
        print(str(epochs_without_improvment) + "ES", end=' ')
        # testen ob training abgebrochen werden kann
        if lr < 0.000001:  # lr == 0 -> keine veränderung -> weitertrainieren ist zeitverschwendung
            history["trainstop"] = ["lr = " + str(lr)]
            print("learning rate to low, stop training " + saveName)
            break
        if epochs_without_improvment > long_epochs*3:
            history["trainstop"] = ["epochs_without_imporvment = " + str(epochs_without_improvment)]
            print("no imprevment of val_loss, stop training " + saveName)
            break
        if epochcount >= 1000:
            history["trainstop"] = ["epochcount = " + str(epochcount)]
            print("end of loop reached, stop training " + saveName)
            break
    dt = time.time() - start_time
    history["trainingtime"] = [dt]
    print("\n", saveName, " took ", dt, "s to fit")

    # print("len(history[loss]) = ", len(history['loss']))
    # print("len(history[val_loss]) = ", len(history['val_loss']))
    # print("len(history[lr) = ", len(history['lr']))
    # print("epochcount = ", epochcount)
    model.set_weights(best_model[1])
    model.save(Dataloader.models_dir + saveName + ".h5")
    save_dict(history, saveName)
    # show_trainhistory(history, saveName)
    return history


def infer(name, dataset):
    test_x, test_y = dataset.get_batch(10)
    model = tf.keras.models.load_model(Dataloader.models_dir + name + ".h5")
    config = model.get_config()  # Returns pretty much every information about your model
    input_size = config["layers"][0]["config"]["batch_input_shape"][1:]  # returns a tuple of width, height and channels
    # input_size = model.layers[0].input_shape[0][1:]  # for test.h5 from findfollowreadlite_dense
    print("infe.input_size: ", input_size)
    for img in test_x:
        # test that all images have the correct size for the tf.model
        assert img.shape[0] == input_size[0] and img.shape[1] == input_size[1]
    # [(assert img.shape == input_size) for img in test_x] why, python? I just want to use assert statements in list comprehensions. is that to much?
    infered = []
    for img in [np.reshape(img, (1, img.shape[0], img.shape[1])) for img in test_x]:
        if img.shape[0] > input_size[0] or img.shape[1] > input_size[1]:
            print("validation image has to be the same size or smaler than largest training image")
            return None
        points = model.predict([img])  # returns list of predictions.
        # print("infer: ", img, " -> ", points)
        #print("   infer: ", points)
        infered.append(points[0])
        # img is of type float, cv2 needs type int to show.
    # test_x = dataset.show((test_x, test_y))
    dataset.show((test_x, test_y), infered)

    # test_x = [np.pad(img, ((0, input_size[0]-img.shape[0]), (0, input_size[1]-img.shape[1])), mode='constant', constant_values=255) for img in test_x]
    # print("test_x: ", type(test_x[0]))
    # print("test_y: ", type(test_y[0]))
    # loss, acc = model.evaluate(test_x, test_y, verbose=2)
    # print("model, accuracy: {:5.2f}%".format(100 * acc))


def test_model(name, dataset):
    test_x, test_y = dataset.get_batch(32 ** 2)
    model = tf.keras.models.load_model(Dataloader.models_dir + name + ".h5")
    config = model.get_config()  # Returns pretty much every information about your model
    input_size = config["layers"][0]["config"]["batch_input_shape"][1:]  # returns a tuple of width, height and channels
    # input_size = model.layers[0].input_shape[0][1:]  # for test.h5 from findfollowreadlite_dense
    print("infe.input_size: ", input_size)
    for img in test_x:
        # test that all images have the correct size for the tf.model
        assert img.shape[0] == input_size[0] and img.shape[1] == input_size[1]
    infered = []
    for img in [np.reshape(img, (1, img.shape[0], img.shape[1])) for img in test_x]:
        if img.shape[0] > input_size[0] or img.shape[1] > input_size[1]:
            print("validation image has to be the same size or smaler than largest training image")
            return None
        points = model.predict([img])  # returns list of predictions.
        # print(img, " -> ", points)
        infered.append(points[0])
    score = 0
    for i in range(len(test_y)):
        gl = Dataloader.dense2points(test_y[i])
        pred = Dataloader.dense2points(infered[i])
        score += sum(abs(gl[t][0] - pred[t][0]) + abs(gl[t][0] - pred[t][0]) for t in range(len(gl)))
    return score


if __name__ == "__main__":
    print("start")
    # TODO Montag email fortschritt
    dataset = Dataloader.Dataset(Dataloader.data_dir, Dataloader.goldlabel_types.linepositions, Dataloader.goldlabel_encodings.dense)
    #dataset = Dataloader.Dataset_test(3)
    #dataset.show(dataset.get_batch(20))
    #exit(0)
    x, y = dataset.get_batch(1024)  # validation data
    x_shape = x[0].shape
    print(x_shape, " -> ", y[0].shape)
    scores = []
    activations = ["softmax", "elu", "exponential", "gelu", "hard_sigmoid", "linear", "relu", "selu",
                   "sigmoid", "softmax", "softplus", "softsign", "swish", "tanh"]
    models = [Models.fullyconnectedFedforward, Models.cvff, Models.conv, Models.vgg11]
    # for (modelf, act, start_lr) in [(Models.conv, "linear", 8), (Models.fullyconnectedFedforward, "linear", 11)]:
    for modelf in models:
        for act in ["hard_sigmoid", "swish", "linear"]:
            for start_lr in [8]:
                name = dataset.name + "_" + str(modelf.__name__) + str(act) + "_mse_lr" + str(start_lr)
                print("name: ", name)

                model = modelf(in_shape=(x_shape[0], x_shape[1], 1), out_length=y[0].shape[0], activation=act)
                model.summary()
                train(model, name, dataset, val=(x, y), start_lr=2**(-start_lr))
                #infer(name, dataset)

                #grade = test_model(name, dataset)
                #scores.append((name, grade))

                history = read_dict(name)
                show_trainhistory(history, name)
    scores.sort(key=lambda x: x[1])
    print("scores =", scores)

scores = [('test3_convhard_sigmoid_mse_lr8', 4940), ('test3_convswish_mse_lr8', 5398), ('test3_convlinear_mse_lr8', 5446), ('test3_convelu_mse_lr8', 5760), ('test3_convlinear_mse_lr8', 5776), ('test3_convsigmoid_mse_lr8', 6076), ('test3_convrelu_mse_lr8', 6094), ('test3_convsoftplus_mse_lr8', 6160), ('test3_convselu_mse_lr8', 6322), ('test3_cvffrelu_mse_lr8', 6504), ('test3_cvffselu_mse_lr8', 6792), ('test3_convsoftsign_mse_lr8', 6878), ('test3_convexponential_mse_lr8', 7382), ('test3_convtanh_mse_lr8', 16024),
          ('test3_cvffhard_sigmoid_mse_lr8', 6968), ('test3_cvffswish_mse_lr8', 7006), ('test3_cvffsoftsign_mse_lr8', 7950), ('test3_cvffelu_mse_lr8', 7968), ('test3_cvfftanh_mse_lr8', 7994), ('test3_cvffgelu_mse_lr8', 8128), ('test3_cvfflinear_mse_lr8', 8408), ('test3_cvfflinear_mse_lr8', 8454), ('test3_cvffsoftplus_mse_lr8', 8540), ('test3_cvffsigmoid_mse_lr8', 8770), ('test3_cvffexponential_mse_lr8', 10524),
          ('test3_fullyconnectedFedforwardsigmoid_mse_lr8', 11796), ('test3_fullyconnectedFedforwardhard_sigmoid_mse_lr8', 11924), ('test3_fullyconnectedFedforwardrelu_mse_lr8', 13224), ('test3_fullyconnectedFedforwardsoftsign_mse_lr8', 14304), ('test3_fullyconnectedFedforwardgelu_mse_lr8', 15942), ('test3_fullyconnectedFedforwardswish_mse_lr8', 17210), ('test3_fullyconnectedFedforwardexponential_mse_lr8', 17358), ('test3_fullyconnectedFedforwardselu_mse_lr8', 18026), ('test3_fullyconnectedFedforwardlinear_mse_lr8', 18436), ('test3_fullyconnectedFedforwardlinear_mse_lr8', 18458), ('test3_fullyconnectedFedforwardelu_mse_lr8', 19732), ('test3_convgelu_mse_lr8', 20350), ('test3_fullyconnectedFedforwardtanh_mse_lr8', 24462), ('test3_fullyconnectedFedforwardsoftplus_mse_lr8', 24556), ('test3_fullyconnectedFedforwardsoftmax_mse_lr8', 31436), ('test3_fullyconnectedFedforwardsoftmax_mse_lr8', 33648), ('test3_convsoftmax_mse_lr8', 33742), ('test3_cvffsoftmax_mse_lr8', 33784), ('test3_convsoftmax_mse_lr8', 33986), ('test3_cvffsoftmax_mse_lr8', 35320)]


import cv2
import keras.losses
import numpy as np
import tensorflow as tf
from keras import backend
import time

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
    try:
        import matplotlib.pyplot as ploter
    except:
        print("Matplotlib is not installed, cant execute show_trainhistory")
        return None
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


def train(model, saveName, dataset, val=None, start_lr=2**(-10), batch_size=None, max_data_that_fits_in_memory=100000):
    #TODO batch size < len(x_train), x_train alle Daten
    start_time = time.time()
    # find better batch_size
    #init other values
    if val is None or len(val) == 0:
        val = dataset.get_batch(128)
    # print("train: ", x_train[0], " -> ", y_train[0])
    #print("x.shape = ", val[0][0].shape)
    #print("y.shape = ", val[1][0].shape)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=30, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    lr = start_lr
    lr_mult = [0.5, 1, 2]  # at every step: train with all [lr*m for m in lr_mult] learning rates, pick the weights and lr that has the best validation_loss
    print("learning rate: ", end=' ')
    history = {"loss": [], "val_loss": [], "lr": []}
    old_lr = -1
    older_lr = -2
    epochs_without_improvment = 0
    valLoss = 1
    x_train, y_train = dataset.get_batch(max_data_that_fits_in_memory)
    train_tfds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    epochcount = 0
    long_epochs = 64
    short_epochs = 2
    best_model = (valLoss, model.get_weights())

    # train
    while True:
        if lr == old_lr or lr == older_lr:
            # train for long
            print(str(lr) + "L", end=' ')
            backend.set_value(model.optimizer.learning_rate, lr)
            history_next = model.fit(x=train_tfds, epochs=long_epochs, callbacks=[callback], validation_data=val, verbose=0).history
            del x_train
            del y_train
            old_lr = -1
            older_lr = -2
            x_train, y_train = dataset.get_batch(batch_size)
            train_tfds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        else:
            # train with different learning rates.
            # print("test lrs = ", [lr*m for m in lr_mult])
            weights_pre = model.get_weights()
            weigths_post = [({}, [])] * len(lr_mult)
            print(str(lr) + "T", end=' ')
            for i in range(len(lr_mult)):
                backend.set_value(model.optimizer.learning_rate, lr * lr_mult[i])
                tmphistory = model.fit(x=train_tfds, epochs=short_epochs, validation_data=val, verbose=0)
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
        if lr < 0.00000001:  # lr == 0 -> keine veränderung -> weitertrainieren ist zeitverschwendung
            history["trainstop"] = ["lr = " + str(lr)]
            print("learning rate to low, stop training " + saveName)
            break
        if epochs_without_improvment > long_epochs*8:
            history["trainstop"] = ["epochs_without_imporvment = " + str(epochs_without_improvment)]
            print("no imprevment of val_loss, stop training " + saveName)
            break
        if epochcount >= 16000:
            history["trainstop"] = ["epochcount = " + str(epochcount)]
            print("end of loop reached, stop training " + saveName)
            break
    dt = time.time() - start_time
    history["trainingtime"] = [dt]
    print("\n", saveName, " took ", dt, "s to fit to val_loss of ", best_model[0])

    # print("len(history[loss]) = ", len(history['loss']))
    # print("len(history[val_loss]) = ", len(history['val_loss']))
    # print("len(history[lr) = ", len(history['lr']))
    # print("epochcount = ", epochcount)
    model.set_weights(best_model[1])
    model.save(Dataloader.models_dir + saveName + ".h5")
    save_dict(history, saveName)
    show_trainhistory(history, saveName)
    return history


def train2(model, saveName, dataset, val=None, start_lr=2**(-8), batch_size=16):
    start_time = time.time()
    if val is None:
        val = dataset.get_batch(32)
        val = tf.data.Dataset.from_tensor_slices((val[0], val[1])).batch(len(val[0]))

    max_data_that_fits_in_memory = 100000  # sollte von größe der Daten und des freien Speichers abhängen


    # train is Batchdataset with elements of shape (batch_size:<x_train[0].shape, y_train[0].shape>)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    history = {"loss": [], "val_loss": [], "lr": []}

    for lr in [start_lr]:  # [start_lr*(2**-i) for i in range(4)]:
        backend.set_value(model.optimizer.learning_rate, lr)
        x_train, y_train = dataset.get_batch(max_data_that_fits_in_memory)
        train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        del x_train, y_train
        tmphistory = model.fit(x=train, epochs=128, callbacks=[callback], validation_data=val, verbose=1).history
        del train
        history["loss"] += tmphistory["loss"]
        history["val_loss"] += tmphistory["val_loss"]
        history["lr"] += [lr]*len(tmphistory["loss"])

    dt = time.time() - start_time
    history["trainingtime"] = [dt]
    print("\n", saveName, " took ", dt, "s to fit to val_loss of ", min(history["val_loss"]))
    model.save(Dataloader.models_dir + saveName + ".h5")
    save_dict(history, saveName)
    show_trainhistory(history, saveName)
    return history

def infer(name, dataset):
    test_x, test_y = dataset.get_batch(10)
    model = tf.keras.models.load_model(Dataloader.models_dir + name + ".h5")
    config = model.get_config()  # Returns pretty much every information about your model
    input_size = config["layers"][0]["config"]["batch_input_shape"][1:]  # returns a tuple of width, height and channels
    # input_size = model.layers[0].input_shape[0][1:]  # for test.h5 from findfollowreadlite_dense
    print("main.infer: input_size: ", input_size)
    for img in test_x:
        # test that all images have the correct size for the tf.model
        if not (img.shape[0] == input_size[0] and img.shape[1] == input_size[1]):
            print("img shape = ", img.shape, " and input shape = ", input_size, " should be the same")
        assert img.shape[0] == input_size[0] and img.shape[1] == input_size[1]
    # [(assert img.shape == input_size) for img in test_x] why, python? I just want to use assert statements in list comprehensions. is that to much?
    infered = []
    for img in [np.reshape(img, (1, img.shape[0], img.shape[1])) for img in test_x]:
        #if img.shape[0] > input_size[0] or img.shape[1] > input_size[1]:
        #    print("img shape = ", img.shape, " and input shape = ", input_size, " should be the same")
        #    print("validation image has to be the same size or smaler than largest training image")
        #    return None
        pred = model.predict([img])  # returns list of predictions.
        if len(infered) == 0:
            print("main.infer: ", Dataloader.getType(img), " -> ", Dataloader.getType(pred))
        print("main.infer: pred = ", ' '.join(str(pred).replace("\t", " ").replace("\n", " ").split()))
        shape = tuple([x for x in np.array(pred).shape if x > 1])
        pred = np.reshape(pred, shape)
        infered.append(pred)
        # img is of type float, cv2 needs type int to show.
    # test_x = dataset.show((test_x, test_y))
    dataset.show((test_x, test_y), infered)

    # test_x = [np.pad(img, ((0, input_size[0]-img.shape[0]), (0, input_size[1]-img.shape[1])), mode='constant', constant_values=255) for img in test_x]
    # print("test_x: ", type(test_x[0]))
    # print("test_y: ", type(test_y[0]))
    # loss, acc = model.evaluate(test_x, test_y, verbose=2)
    # print("model, accuracy: {:5.2f}%".format(100 * acc))


def show_models(model, name, dataset):
    # https://www.tensorflow.org/tensorboard/get_started
    # https://www.tensorflow.org/tensorboard/graphs
    # tensorboard --logdir C:\Users\Idefix\PycharmProjects\SimpleHTR\data\modelgraph\
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=Dataloader.data_dir+"modelgraph/"+name)
    x, y = dataset.get_batch(10)
    model.fit(x, y, batch_size=1, epochs=1, callbacks=[tensorboard_callback])


    linedetection_models = [("fcff", Models.fullyconnectedFedforward), ("fcff2", Models.fullyconnectedFedforward2), ("cvff", Models.cvff), ("conv", Models.conv), ("vgg11", Models.vgg11)]
    dataset = Dataloader.Dataset(gl_type=Dataloader.GoldlabelTypes.linepositions, img_type=Dataloader.ImgTypes.paragraph)
    x, y = dataset.get_batch(10)
    in_shape = (x[0].shape[0], x[0].shape[1], 1)
    out_length = len(y[0])
    print(in_shape, " -> ", out_length)
    for (mn, mf) in linedetection_models:
        print("name = ", mn)
        model = mf(in_shape=in_shape, out_length=out_length, activation="hard_sigmoid")
        model.summary()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=Dataloader.data_dir+"modelgraph/"+mn)
        x, y = dataset.get_batch(10)
        model.fit(x, y, batch_size=1, epochs=1, callbacks=[tensorboard_callback])
        print("\n\n")
    readline_models = [("simpleHTR", Models.simpleHTR), ("simpleHTR2", Models.simpleHTR2)]
    dataset = Dataloader.Dataset(gl_type=Dataloader.GoldlabelTypes.text, img_type=Dataloader.ImgTypes.line)
    for (mn, mf) in readline_models:
        print("name = ", mn)
        model = mf(in_shape=(32, 128, 1), out_length=27, activation="hard_sigmoid")
        model.summary()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=Dataloader.data_dir+"modelgraph/"+mn)
        x, y = dataset.get_batch(10)
        model.fit(x, y, batch_size=1, epochs=1, callbacks=[tensorboard_callback])
    exit(0)


def external_seg(modelname_lp, modelname_htr, ds_ptxt):
    model_lp = tf.keras.models.load_model(Dataloader.models_dir + modelname_lp + ".h5")
    model_htr = tf.keras.models.load_model(Dataloader.models_dir + modelname_htr + ".h5")
    paraimg_size = model_lp.get_config()["layers"][0]["config"]["batch_input_shape"][1:]
    imgs, texts = ds_ptxt.get_batch(10)
    print("main.external_seg: imgs = ", Dataloader.getType(imgs))
    for img in imgs:
        linepoints = model_lp.predict([np.reshape(img, (1, img.shape[0], img.shape[1]))])
        print("main.external_seg: pred = ", Dataloader.getType(linepoints))
        linepoints = np.array(linepoints[0])
        lineimgs = [Dataloader.extractline(img, linepoints[i*5:(i+1)*5]) for i in range(len(linepoints)//5)]
        print("main.external_seg: lineimgs = ", Dataloader.getType(lineimgs))
        texts = [model_htr.predict([np.reshape(limg, (1, limg.shape[0], limg.shape[1]))])[0] for limg in lineimgs]
        for txt in texts:
            print("pred_txt = ", Dataloader.sparse2txt(txt))
        cv2.imshow("paragrph img", np.array(img, dtype="uint8"))
        [cv2.imshow("line"+str(i), lineimgs[i]) for i in range(len(lineimgs))]
        cv2.waitKey(0)
    return None


if __name__ == "__main__":
    # nach linefinder paralelisieren, dann mit FC zu num_lines*char_per_line*(num_chars+blank+linebreak) umwandeln
    # 0. Readme
    # 0. model_lp auf echten daten funktionieren machen
    # 0. zeilen und word/zeile an echte Daten anpassen, model_lp dadrauf trainieren.
    # neue NN ansätze:
    # 1. bild as [[pixel]], mit pixel = (color, maske)
    # 2. htr3: paragrph, lp -> flatten(conv(maxpooling(paragrph))+lp
    # 3. htr: bild mit höhe = 5+zeilenhöhe -> tensor(5, zeilenlänge) ->  tensor(5, linepoint+zeilenlänge)
    # 4. segmentation durch matrixmult lösen
    # sonstiges
    # 1. ParagrphBilder als ausschnitt aus größerem Bild
    # 2. Zeilenhöhe wie zeilenwinkel festlegen
    # 4. variable länge
    # 5. zeilen beim ausschneiden linepoint auf schnittpunkt
    # ENDZIEL: echte Daten von Gold auslesen
    # lineRecognition
    # batch-normalisation als attention  # https://github.com/Nikolai10/scrabble-gan
    print("\n\n"+"-"*64+"start"+"-"*64+"\n\n")
    # init all datasets needed.
    #external_seg("lp_conv", "htr", Dataloader.Dataset(img_type=Dataloader.ImgTypes.paragraph, gl_type=Dataloader.GoldlabelTypes.text))
    ds_plp = Dataloader.Dataset(img_type=Dataloader.ImgTypes.paragraph, gl_type=Dataloader.GoldlabelTypes.linepositions)
    #ds_plimg = Dataloader.Dataset(img_type=Dataloader.ImgTypes.paragraph, gl_type=Dataloader.GoldlabelTypes.lineimg)
    #ds_ptxt = Dataloader.Dataset(img_type=Dataloader.ImgTypes.paragraph, gl_type=Dataloader.GoldlabelTypes.text)
    #ds_ltxt = Dataloader.Dataset(img_type=Dataloader.ImgTypes.line, gl_type=Dataloader.GoldlabelTypes.text)

    model = Models.conv2(in_shape=ds_plp.imgsize, out_length=ds_plp.glsize, inner_activation="relu", activation="hard_sigmoid")
    batch_size = 32
    maxdata = 100000
    savename = f"{ds_plp.name}_{model.name}_relu_hard_sigmoid_t1tfds{maxdata}_{batch_size}"
    print("train ", savename)
    train(model, savename, ds_plp, batch_size=batch_size, max_data_that_fits_in_memory=maxdata)
    exit(0)

    if False:  # test Dataloader.extractline
        ds = Dataloader.Dataset(img_type=Dataloader.ImgTypes.paragraph, gl_type=Dataloader.GoldlabelTypes.linepositions)
        batch = ds.get_batch(10)
        for i in range(len(batch[0])):
            (img, lp) = batch[0][i], batch[1][i]
            print("(img, lp) = ", Dataloader.getType((img, lp)))
            ds.show(([img], [lp]))
            limgs = [Dataloader.extractline(img, lp[il*5:(il+1)*5]) for il in range(len(lp)//5)]
            [cv2.imshow("line"+str(li), np.array(limgs[li], dtype="uint8")) for li in range(len(limgs))]
            cv2.waitKey(0)
        exit(0)

    for savename in ["Dataset_real(22, 1)_conv2_relu_hard_sigmoid_t2tfds", "Dataset_real(22, 1)_conv_relu_hard_sigmoid_t2tfds"]:  # ["test2_cvff_tanh_hard_sigmoid"]:  # ["Dataset_real(22, 1)_cvff_relu_hard_sigmoid0", "Dataset_real(22, 1)_conv_relu_hard_sigmoid0"]:
        print("infer: "+savename)
        #history = read_dict(savename)
        #save_dict(history, savename)
        #show_trainhistory(history, savename)
        infer(savename, ds_plp)

    #train all relevant models
    # training one model works fine.
    # training multiply models sequentially throus OOM. https://stackoverflow.com/questions/42886049/keras-tensorflow-cpu-training-sequential-models-in-loop-eats-memory

    # linepoint
    for modelf in [Models.conv, Models.conv2]:
        for inner_activation in ["relu"]:  # , "relu", "elu", "gelu", "hard_sigmoid", "selu", "sigmoid", "swish"]:
            final_activation = "hard_sigmoid"
            print("start training "+savename)
            model = modelf(in_shape=ds_plp.imgsize, out_length=ds_plp.glsize, activation=final_activation, loss=keras.losses.MeanSquaredError(), inner_activation=inner_activation)
            savename = f"{ds_plp.name}_{model.name}_{inner_activation}_{final_activation}_t2tfds"
            train2(model, saveName=savename, dataset=ds_plp, batch_size=4)
            del model  # model parameters gets saved by train
            tf.keras.backend.clear_session()
            print("finished training "+savename)
    exit(0)

    #htr
    losses = [(Models.CTCLoss_issus(logit_length=ds_ltxt.imgsize[1]//4, blank_index=0), "_ctc"), (keras.losses.MeanSquaredError(), "_mse")]  # TODO CTC loss not working
    for (loss, nm) in losses:
        model_htr = Models.htr(in_shape=ds_ltxt.imgsize, loss=loss)
        train(model_htr, saveName="htr"+nm, dataset=ds_ltxt, batch_size=64)
    print("finished training htr")
    exit(0)


    # linefinder
    #linesshape = (maxlinecount, ds_ltxt.imgsize[0], ds_ltxt.imgsize[1])
    # Process finished with exit code -1073740791 (0xC0000409)
    # while SSD-auslastung == 100%, arbeitsspeicherauslastung > 90%, cpu-auslastung zwieschen 10% und 100%
    # inklusive PC-freeze für 1min
    # das alles in der model.fit methode (in main.train(model_linefinder))
    model_linefinder = Models.linefinder(in_shape=ds_plimg.imgsize, output_shape=linesshape)
    #model_linefinder.summary()  # 259,550 params < conv(256, 256)
    train(model_linefinder, saveName="linefinder", dataset=ds_plimg)

    #linefinder2
    #model_conv = Models.conv(in_shape=ds_plp.imgsize, out_length=maxlinecount*5)
    #train(model_conv, saveName="lp_conv", dataset=ds_plp, start_lr)
    #model_linefinder2 = Models.linefinder2(model_conv, out_shape=linesshape)
    #train(model_linefinder2, saveName="limg_linefinder2", dataset=ds_plimg, start_lr=0.0000001)
    #del model_conv
    #del model_linefinder2

    #total
    model_total = Models.total(linefindermodel=model_linefinder, linereadermodel=model_htr)
    train(model_total, saveName="total", dataset=ds_ptxt, start_lr=0.0000001)
    exit(0)
    #names = ["real_convhard_sigmoid_mse_lr8", "real_convswish_mse_lr8", "real_cvfflinear_mse_lr8", "real_vgg11hard_sigmoid_mse_lr8"]  # scheinen zu funktionieren
    #for n in names:
    #    print("\n", n)
    #    infer(n, dataset)
    exit(0)

    dataset = Dataloader.Dataset(img_type=Dataloader.ImgTypes.paragraph, gl_type=Dataloader.GoldlabelTypes.linepositions)
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
                #model = modelf(in_shape=(x_shape[0], x_shape[1], 1), out_length=y[0].shape[0], activation=act)
                #model.summary()
                #train(model, name, dataset, val=(x, y), start_lr=2**(-start_lr))
                infer(name, dataset)

                #grade = test_model(name, dataset)
                #scores.append((name, grade))

                #history = read_dict(name)
                #show_trainhistory(history, name)
    scores.sort(key=lambda x: x[1])
    print("scores =", scores)

scores = [('real_convhard_sigmoid_mse_lr8', 5954), ('real_cvfflinear_mse_lr8', 6660), ('real_convswish_mse_lr8', 7492), ('real_vgg11hard_sigmoid_mse_lr8', 9320), ('real_fullyconnectedFedforwardswish_mse_lr8', 41178), ('real_cvffhard_sigmoid_mse_lr8', 42088), ('real_vgg11swish_mse_lr8', 42398), ('real_vgg11linear_mse_lr8', 43094), ('real_cvffswish_mse_lr8', 43234), ('real_convlinear_mse_lr8', 43262), ('real_fullyconnectedFedforwardlinear_mse_lr8', 43804), ('real_fullyconnectedFedforwardhard_sigmoid_mse_lr8', 72168)]

scores = [('test3_convhard_sigmoid_mse_lr8', 4940), ('test3_convswish_mse_lr8', 5398), ('test3_convlinear_mse_lr8', 5446), ('test3_convelu_mse_lr8', 5760), ('test3_convlinear_mse_lr8', 5776), ('test3_convsigmoid_mse_lr8', 6076), ('test3_convrelu_mse_lr8', 6094), ('test3_convsoftplus_mse_lr8', 6160), ('test3_convselu_mse_lr8', 6322), ('test3_cvffrelu_mse_lr8', 6504), ('test3_cvffselu_mse_lr8', 6792), ('test3_convsoftsign_mse_lr8', 6878), ('test3_convexponential_mse_lr8', 7382), ('test3_convtanh_mse_lr8', 16024),
          ('test3_cvffhard_sigmoid_mse_lr8', 6968), ('test3_cvffswish_mse_lr8', 7006), ('test3_cvffsoftsign_mse_lr8', 7950), ('test3_cvffelu_mse_lr8', 7968), ('test3_cvfftanh_mse_lr8', 7994), ('test3_cvffgelu_mse_lr8', 8128), ('test3_cvfflinear_mse_lr8', 8408), ('test3_cvfflinear_mse_lr8', 8454), ('test3_cvffsoftplus_mse_lr8', 8540), ('test3_cvffsigmoid_mse_lr8', 8770), ('test3_cvffexponential_mse_lr8', 10524),
          ('test3_fullyconnectedFedforwardsigmoid_mse_lr8', 11796), ('test3_fullyconnectedFedforwardhard_sigmoid_mse_lr8', 11924), ('test3_fullyconnectedFedforwardrelu_mse_lr8', 13224), ('test3_fullyconnectedFedforwardsoftsign_mse_lr8', 14304), ('test3_fullyconnectedFedforwardgelu_mse_lr8', 15942), ('test3_fullyconnectedFedforwardswish_mse_lr8', 17210), ('test3_fullyconnectedFedforwardexponential_mse_lr8', 17358), ('test3_fullyconnectedFedforwardselu_mse_lr8', 18026), ('test3_fullyconnectedFedforwardlinear_mse_lr8', 18436), ('test3_fullyconnectedFedforwardlinear_mse_lr8', 18458), ('test3_fullyconnectedFedforwardelu_mse_lr8', 19732), ('test3_convgelu_mse_lr8', 20350), ('test3_fullyconnectedFedforwardtanh_mse_lr8', 24462), ('test3_fullyconnectedFedforwardsoftplus_mse_lr8', 24556), ('test3_fullyconnectedFedforwardsoftmax_mse_lr8', 31436), ('test3_fullyconnectedFedforwardsoftmax_mse_lr8', 33648), ('test3_convsoftmax_mse_lr8', 33742), ('test3_cvffsoftmax_mse_lr8', 33784), ('test3_convsoftmax_mse_lr8', 33986), ('test3_cvffsoftmax_mse_lr8', 35320)]


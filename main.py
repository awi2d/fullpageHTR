import cv2
import numpy as np
import tensorflow as tf
from keras import backend
import time


import Dataloader
import Models

# <debug functions>

def save_dict(history, name="unnamed_model"):
    #assert len(history["loss"]) == len(history["val_loss"]) == len(history["lr"])
    with open(Dataloader.models_dir+str(name)+".txt", 'w') as f:
        for k in history.keys():
            for e in history[k]:
                print(e, file=f)
            print(k, file=f)

def read_dict(name):
    with open(Dataloader.models_dir+str(name)+".txt", 'r') as f:
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
    fig, host = ploter.subplots(figsize=(15,10)) # (width, height) in inches
    # (see https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.subplots.html)

    par1 = host.twinx()

    host.set_xlim(0, len(history["loss"]))
    host.set_ylim(0, max(max(history["loss"]), max(history["val_loss"])))
    par1.set_ylim(0, max(history["lr"]))

    host.set_xlabel("Epochs")
    host.set_ylabel("loss")
    par1.set_ylabel("learning_rate")

    color1 = ploter.cm.viridis(0)
    color2 = ploter.cm.viridis(0.5)
    color3 = ploter.cm.viridis(.9)

    xtmp = list(range(len(history["loss"])))
    p1, = host.plot(xtmp, history["loss"], color=color1, label="training loss")
    p2, = host.plot(xtmp, history["val_loss"], color=color2, label="validation loss")
    p3, = par1.plot(xtmp, history["lr"], color=color3, label="learning rate")

    lns = [p1, p2, p3]
    host.legend(handles=lns, loc='best')

    # right, left, top, bottom
    #par1.spines['right'].set_position(('outward', 60))

    # Sometimes handy, same for xaxis
    #par2.yaxis.set_ticks_position('right')

    # Move "Velocity"-axis to the left
    # par2.spines['left'].set_position(('outward', 60))
    # par2.spines['left'].set_visible(True)
    # par2.yaxis.set_label_position('left')
    # par2.yaxis.set_ticks_position('left')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    #par2.yaxis.label.set_color(p3.get_color())

    # Adjust spacings w.r.t. figsize
    fig.tight_layout()
    # Alternatively: bbox_inches='tight' within the plt.savefig function
    #                (overwrites figsize)

    # Best for professional typesetting, e.g. LaTeX
    ploter.savefig(Dataloader.models_dir+"trainhistory_"+name+".pdf")
    # For raster graphics use the dpi argument. E.g. '[...].png", dpi=200)'


def show_text_data(training_data):
    for (img, txt) in training_data:
        cv2.imshow(txt, img)
    cv2.waitKey(0)


def show_points_data(training_data, decoding_func):
    h, w = training_data[0][0].shape
    print("main.show_points_data: w, h = ", h, ", ", w)
    if max([max(p) for (img, p) in training_data]) > 1.1 or min([min(p) for (img, p) in training_data]) < -0.1:
        print("main.show_points_data: point outside of bounds. points = ", [points for (img, points) in training_data])
    for (img, points) in training_data:
        img = np.array(img, dtype="uint8")
        print("show_points_data.img: ", Dataloader.getType(img))
        print("show_points_data.pts: ", Dataloader.getType(points))
        points = decoding_func(points, max_x=w, max_y=h)
        for point in points:
            #print("point = ", point)
            point = ((max(1, int(point[0][0])), max(1, int(point[0][1]))), (max(1, int(point[1][0])), max(1, int(point[1][1]))), max(1, int(point[2])))
            cv2.circle(img, point[0], int(point[2]/2), 125, 2)
            cv2.rectangle(img, (point[0][0]-5, point[0][1]-5), (point[0][0]+5, point[0][1]+5), 125, 2)
            cv2.circle(img, point[1], int(point[2]/2), 125, 2)
            cv2.line(img, point[0], point[1], 125, thickness=1)
        cv2.imshow(str(points), img)
    cv2.waitKey(0)
# </debug functions>


def train(model, saveName, x_train, y_train, val):
    assert len(x_train) == len(y_train)  # TODO does not make sense if x_train is of type tf.dataset
    assert len(x_train) > 0
    print("len(x_train) = ", len(x_train))
    if val is None or len(val) == 0:
        val = (x_train, y_train)
    #print("train: ", x_train[0], " -> ", y_train[0])
    print("x.shape = ", x_train[0].shape)
    print("y.shape = ", y_train[0].shape)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    lr = 2**(-8)
    lr_mult = [0.5, 1, 2]  # at every step: train with all [lr*m for m in lr_mult] learning rates, pick the weights and lr that has the best validation_loss
    start_time = time.time()
    print("learning rate: ", lr, end=' ')
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['lr'] = []
    old_lr = -1
    older_lr = -2
    steps_without_improvment = 0
    valLoss = 1
    for i in range(16):
        #print("step start: lr = ", [lr, old_lr, older_lr])
        weights_pre = model.get_weights()
        weigths_post = [({}, [])]*len(lr_mult)

        if lr == old_lr or lr == older_lr:
            #train for long
            print(str(lr)+"L", end=' ')
            backend.set_value(model.optimizer.learning_rate, lr)
            history_next = model.fit(x_train, y_train, epochs=64, steps_per_epoch=len(x_train), callbacks=[callback], validation_data=val, verbose=0).history
            old_lr = -1
            older_lr = -2
        else:
            # train with different learning rates.
            #print("test lrs = ", [lr*m for m in lr_mult])
            print(str(lr)+"T", end=' ')
            for i in range(len(lr_mult)):
                backend.set_value(model.optimizer.learning_rate, lr*lr_mult[i])
                tmphistory = model.fit(x_train, y_train, epochs=2, steps_per_epoch=len(x_train), validation_data=val, verbose=0)
                #print("history: ", history.history)
                weigths_post[i] = (tmphistory.history, model.get_weights())
                model.set_weights(weights_pre)
            # pick best learning rate.
            best = 0
            for i in range(1, len(lr_mult)):
                if weigths_post[i][0]['val_loss'][-1] < weigths_post[best][0]['val_loss'][-1]:
                    best = i
            model.set_weights(weigths_post[best][1])
            history_next = weigths_post[best][0]
            older_lr = old_lr
            old_lr = lr
            lr = lr*lr_mult[best]

        history['val_loss'] = history['val_loss'] + history_next['val_loss']
        history['loss'] = history['loss'] + history_next['loss']
        history['lr'] = history['lr'] + [lr]*len(history_next['loss'])
        if history['val_loss'][-1] >= valLoss:  # in diesem schritt hat sich nichst verbesser
            steps_without_improvment += 1
        else:
            steps_without_improvment = 0
        print(str(steps_without_improvment)+"ES", end=' ')
        valLoss = history['val_loss'][-1]
        # testen ob training abgebrochen werden kann
        if lr < 0.000001:  # lr == 0 -> keine veränderung -> weitertrainieren ist zeitverschwendung
            print("learning rate to low, stop training")
            break
        if steps_without_improvment > 2:
            print("no imprevment of val_loss, stop training")
            break
        if i == 15:
            print("end of loop reached, stop training")
    dt = time.time()-start_time
    print("\n", saveName, " took ", dt, "s to fit")

    #start_time = time.time()
    #model.fit(x_train, y_train, epochs=64, steps_per_epoch=len(x_train), verbose=0)
    #dt = time.time()-start_time
    #print(saveName, " took ", dt, "s to fit\n")
    # dir = Dataloader.data_dir+"/models/"
    model.save(Dataloader.models_dir+saveName+".h5")
    save_dict(history, saveName)
    #show_trainhistory(history, saveName)
    return history


""" 
kommentar:

                
self.history = self.autoencoder_model.fit(tf.data.Dataset.zip((x_train, y_train)),  # x_train und y_train ist tf.Dataset
                                          batch_size=self.batch_size,
                                          epochs=self.n_epochs,
                                          validation_data=tf.data.Dataset.zip((x_val, y_val)),
                                          callbacks=[early_stopping])
"""



def infer(name, decode_func=Dataloader.sparse2linepoints):
    (x, y), (val_x, val_y), (test_x, test_y) = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.dense)
    model = tf.keras.models.load_model("C:/Users/Idefix/PycharmProjects/data/"+name+".h5")
    config = model.get_config()  # Returns pretty much every information about your model
    input_size = config["layers"][0]["config"]["batch_input_shape"][1:]  # returns a tuple of width, height and channels
    #output_size = config["layers"][-1]["config"]["batch_input_shape"][1:]  # TODO dont returns a tuple of width, height and channels
    #input_size = model.layers[0].input_shape[0][1:]  # for test.h5 from findfollowreadlite_dense
    print("input_size: ", input_size)
    #print("output_size: ", output_size)
    img = test_x[0]

    print("img_shape: ", img.shape)
    if img.shape[0] > input_size[0] or img.shape[1] > input_size[1]:
        print("validation image has to be the same size or smaler than largest training image")
        return None
    #img = np.pad(img, ((0, input_size[0]-img.shape[0]), (0, input_size[1]-img.shape[1])), mode='constant', constant_values=255)
    img_reshaped = np.reshape(img, (1, img.shape[0], img.shape[1]))
    print("img_shape: ", Dataloader.getType(img_reshaped))
    points = model.predict([img_reshaped])[0]  # returns list of predictions.
    print("predicted_point = ", Dataloader.getType(points))
    # img is of type float, cv2 needs type int to show.
    img = img.astype(np.uint8)
    #print("img = ", img)
    #points = points[0]  # for test.h5 from findfollowreadlite_dense
    show_points_data([(img, points)], decoding_func=decode_func)

    #test_x = [np.pad(img, ((0, input_size[0]-img.shape[0]), (0, input_size[1]-img.shape[1])), mode='constant', constant_values=255) for img in test_x]
    #print("test_x: ", type(test_x[0]))
    #print("test_y: ", type(test_y[0]))
    #loss, acc = model.evaluate(test_x, test_y, verbose=2)
    #print("model, accuracy: {:5.2f}%".format(100 * acc))


#if __name__ == "__main__":
    #data = Dataloader.getData(dir=Dataloader.data_dir, img_type=Dataloader.img_types.paragraph, goldlabel_type=Dataloader.goldlabel_types.linepositions, goldlabel_encoding=Dataloader.goldlabel_encodings.dense, maxcount=200)
    #data = Dataloader.get_testdata(Dataloader.goldlabel_encodings.dense)
    #Dataloader.store(data, Dataloader.mydata_dir)
    #exit(0)

    # train real model
    #x, y = Dataloader.test_tfds()
    #print("x data type: ", Dataloader.getType(x))
    #print("y data type: ", Dataloader.getType(y))
    #print("x data: ", x)
    #print("y data: ", y)
    #train_data, val, test = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.onehot)
    #x_shape = x[0].shape  # TypeError: 'BatchDataset' object is not subscriptable
    #model = Models.getModel("findfollowreadlite_mse", (x_shape[0], x_shape[1], 1), y[0].shape[0])
    #train(model, "sparse_mse", x, y, (x, y))
    #print("trained model succesfully")


def test_test():
    tc_dense_mse = tf.keras.models.load_model(Dataloader.models_dir+"tc_dense_mse.h5")
    tc_dense_cce = tf.keras.models.load_model(Dataloader.models_dir+"tc_dense_cce.h5")
    tc_sparse_mse = tf.keras.models.load_model(Dataloader.models_dir+"tc_sparse_mse.h5")
    tc_sparse_cce = tf.keras.models.load_model(Dataloader.models_dir+"tc_sparse_cce.h5")
    imgs_dense = Dataloader.get_testdata(enc=Dataloader.goldlabel_encodings.dense)
    imgs_sparse = Dataloader.get_testdata(enc=Dataloader.goldlabel_encodings.onehot)
    for (model, data, enc, name) in [
        (tc_dense_mse, imgs_dense, Dataloader.dense2point, "dense_mse"),
        (tc_dense_cce, imgs_dense, Dataloader.dense2point, "dense_cce"),
        (tc_sparse_mse, imgs_sparse, Dataloader.sparse2point, "sparse_mse"),
        (tc_sparse_cce, imgs_sparse, Dataloader.sparse2point, "sparse_cce")]:
        error = 0
        for (img, gl) in data:
            points = model.predict([np.reshape(img, (1, 32, 32))])  # returns list of predictions.
            #print("points: ", Dataloader.getType(points), "\n", points)
            pred_point = enc(points[0])
            gl = enc(gl)
            te = abs(pred_point[0]-gl[0])+abs(pred_point[1]-gl[1])
            error += te
            #if te > 10:
            #    print("\nte = ", te)
            #    print("gl, pred = ", gl, ", ", pred_point)
            #    cv2.circle(img, pred_point, 5, 125, thickness=2)
            #    cv2.imshow("predicted: "+str(gl), img)
            #    cv2.waitKey(0)
        print("error of ", name, " = ", error)


def train_test():
    """
    uses the simplier find_the_centor_of_the_point dataset and all models
    """
    # train models on test circle data
    imgs_d = Dataloader.get_testdata(enc=Dataloader.goldlabel_encodings.dense)
    x = np.array([d[0] for d in imgs_d])
    y = np.array([np.array(d[1], dtype=float) for d in imgs_d])
    y_size = y[0].shape[0]
    modle = Models.getModel("findfollowreadlite_mse", (32, 32, 1), y_size)
    hist = train(modle, "tc_dense_mse", x, y, val=None)
    show_trainhistory(hist, "tc_dense_mse")
    print("history of tc_dense_mse: \n   loss: ", hist['loss'], "\n   vall: ", hist['val_loss'], "\n   lr__: ", hist['lr'])
    exit(0)
    modle = Models.getModel("findfollowreadlite_cce", (32, 32, 1), y_size)
    hist = train(modle, "tc_dense_cce", x, y, val=None)
    show_trainhistory(hist, "tc_dense_cce")

    imgs_s = Dataloader.get_testdata(enc=Dataloader.goldlabel_encodings.onehot)
    x = np.array([d[0] for d in imgs_s])
    y = np.array([np.array(d[1], dtype=float) for d in imgs_s])
    y_size = y[0].shape[0]
    modle = Models.getModel("findfollowreadlite_mse", (32, 32, 1), y_size)
    hist = train(modle, "tc_sparse_mse", x, y, val=None)
    print("history of tc_sparse_mse: \n   loss: ", hist['loss'], "\n   vall: ", hist['val_loss'], "\n   lr__: ", hist['lr'])
    show_trainhistory(hist, "tc_sparse_mse")
    modle = Models.getModel("findfollowreadlite_cce", (32, 32, 1), y_size)
    hist = train(modle, "tc_sparse_cce", x, y, val=None)
    show_trainhistory(hist, "tc_sparse_cce")
    print("history of tc_sparse_cce: \n   loss: ", hist['loss'], "\n   vall: ", hist['val_loss'], "\n   lr__: ", hist['lr'])
    exit(0)

    imgs_sparse = Dataloader.get_testdata(enc=Dataloader.point2spares)
    cv2.waitKey(0)
    exit(0)

    train_data, val, test = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.dense)
    x_shape = train_data[0][0].shape
    modle = Models.getModel("findfollowreadlite_dense", (x_shape[0], x_shape[1], 1), train_data[1][0].shape[0])
    train(modle, "test_dense", train_data[0], train_data[1], val)
    #infer("dense_mse", decode_func=Dataloader.dense2points)
    infer("dense_mse", decode_func=Dataloader.goldlabel_encodings.dense)


def train_tmp():
    train_data, val, test = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.dense)  # getTrainingData() = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    x_shape = train_data[0][0].shape
    model = Models.getModel("findfollowreadlite_mse", (x_shape[0], x_shape[1], 1), train_data[1][0].shape[0])
    train(model, "dense_mse", train_data[0], train_data[1], val)
    model = Models.getModel("findfollowreadlite_cce", (x_shape[0], x_shape[1], 1), train_data[1][0].shape[0])
    train(model, "dense_cce", train_data[0], train_data[1], val)

    train_data, val, test = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.onehot)
    x_shape = train_data[0][0].shape
    model = Models.getModel("findfollowreadlite_mse", (x_shape[0], x_shape[1], 1), train_data[1][0].shape[0])
    train(model, "sparse_mse", train_data[0], train_data[1], val)
    model = Models.getModel("findfollowreadlite_cce", (x_shape[0], x_shape[1], 1), train_data[1][0].shape[0])
    train(model, "sparse_cce", train_data[0], train_data[1], val)

# original training
# dense_mse  180s,  1366
# dense_cce  180s, 21803
# sparse_mse 200s,   869
# sparse_cce 200s, 31744

# training with adaptive lr
# dense_mse  309s,  1041
# dense_cce  356s, 21823
# sparse_mse 674s,   393
# sparse_cce 511s, 31744

#tc_dense_mse  took  391.2558476924896 s to fit
#tc_dense_cce  took  409.5375266075134 s to fit
#tc_sparse_mse  took  1147.9333064556122 s to fit


if __name__ == "__main__":
    #history_tc_dense_mse = {
    #    "loss":  [0.011603313498198986, 0.0024386029690504074, 0.0021900644060224295, 0.0009324167622253299, 0.0014230174710974097, 0.0008133778464980423, 0.0006078779697418213, 0.000644009152892977, 0.0006799377733841538, 0.0007162592955864966, 0.000681737728882581, 0.0006335963844321668, 0.000691279536113143, 0.0005902193952351809, 0.0006145625375211239, 0.0005561542930081487, 0.0004105496045667678, 0.00038093404145911336, 0.0002799385110847652, 0.0002644509368110448, 0.00024023311561904848, 0.00022923607320990413, 0.0002127451734850183, 0.00023080526443663985, 0.00021536061831284314, 0.0002125465834978968, 0.0001934827014338225, 0.00019323888409417123, 0.00020716720609925687, 0.00019979229546152055, 0.00019049090042244643, 0.00020356645109131932, 0.00018907342746388167, 0.00020284639322198927, 0.00019228846940677613, 0.0002177925780415535],
    #    "val_loss":  [0.002282543107867241, 0.0009632460423745215, 0.0007131692254915833, 0.0004665470332838595, 0.0004797406436409801, 0.000326928828144446, 0.0005161724402569234, 0.0003961642796639353, 0.0003904254990629852, 0.0007137289503589272, 0.0003142130735795945, 0.0004579871892929077, 0.0003300999815110117, 0.0005500881234183908, 0.00043778124381788075, 0.0004172158078290522, 0.00021023709268774837, 0.00015162666386459023, 0.00010275138629367575, 8.319074549945071e-05, 7.84482472226955e-05, 8.545017044525594e-05, 5.936884554103017e-05, 5.620326192001812e-05, 5.25508112332318e-05, 5.056882946519181e-05, 5.0369519158266485e-05, 4.749029176309705e-05, 4.658548641600646e-05, 4.6408840717049316e-05, 4.477115362533368e-05, 4.5409520680550486e-05, 4.4636690290644765e-05, 4.472305954550393e-05, 4.4687625631922856e-05, 4.447313040145673e-05],
    #    "lr":  [0.001953125, 0.001953125, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.0009765625, 0.00048828125, 0.00048828125, 0.000244140625, 0.000244140625, 0.0001220703125, 0.0001220703125, 6.103515625e-05, 6.103515625e-05, 3.0517578125e-05, 3.0517578125e-05, 1.52587890625e-05, 1.52587890625e-05, 7.62939453125e-06, 7.62939453125e-06, 3.814697265625e-06, 3.814697265625e-06, 1.9073486328125e-06, 1.9073486328125e-06, 9.5367431640625e-07, 9.5367431640625e-07]}
    #save_dict(history_tc_dense_mse, "tc_dense_mse")
    #exit(0)
    train_data, val, test = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.dense)  # getTrainingData() = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    #show_points_data([(train_data[0][i], train_data[1][i]) for i in range(len(train_data[0]))], Dataloader.dense2linepoints)
    x_shape = train_data[0][0].shape
    print("x_shape = ", x_shape)
    model = Models.getModel("findfollowreadlite_mse", (x_shape[0], x_shape[1], 1), train_data[1][0].shape[0])
    train(model, "dense_mse", train_data[0], train_data[1], val)



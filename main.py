import cv2
import numpy as np
import tensorflow as tf
from keras import backend
import time

import Dataloader
import Models

# <debug functions>


def show_data(training_data):
    for (img, txt) in training_data:
        cv2.imshow(txt, img)
    cv2.waitKey(0)


def show_points_data(training_data, decoding_func):
    h, w = training_data[0][0].shape
    print("main.show_points_data: w, h = ", h, ", ", w)
    if max([max(p) for (img, p) in training_data]) > 1.1 or min([min(p) for (img, p) in training_data]) < -0.1:
        print("main.show_points_data: point outside of bounds. points = ", [points for (img, points) in training_data])
    for (img, points) in training_data:
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


def todo_in_dataloader_integrieren():
    data_dir = "C:\\Users\\Idefix\\PycharmProjects\\SimpleHTR\\trainingDataset"
    all_data = Dataloader.getData(dir=data_dir, dataset_loader=Dataloader.dataset_names.iam, img_type=Dataloader.img_types.paragraph, goldlabel_type=Dataloader.goldlabel_types.linepositions, goldlabel_encoding=Dataloader.goldlabel_encodings.dense, maxcount=300, x_size=(1000, 2000))
    #show_points_data(all_data[:10])
    train_test_split = 0.8
    train_test_split = int(len(all_data)*train_test_split)
    val_test_split = int(len(all_data)*0.9)

    training_data = all_data[:train_test_split]
    validation_data = all_data[train_test_split:val_test_split]
    test_data = all_data[val_test_split:]

    train_x = np.array([t[0] for t in training_data])
    train_y = np.array([t[1] for t in training_data])
    val_x   = np.array([t[0] for t in validation_data])
    val_y   = np.array([t[1] for t in validation_data])
    test_x  = np.array([t[0] for t in test_data])
    test_y  = np.array([t[1] for t in test_data])
    return train_x, train_y, val_x, val_y, test_x, test_y


def train(model, saveName, x_train, y_train, val):
    assert len(x_train) == len(y_train)
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
        weigths_post = [0]*len(lr_mult)

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
                # TODO ValueError: `labels.shape` must equal `logits.shape` except for the last dimension. Received: labels.shape=(2,) and logits.shape=(1, 2)
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
        valLoss = history['val_loss'][-1]
        # testen ob training abgebrochen werden kann
        if lr < 0.00000001:  # lr == 0 -> keine veränderung -> weitertrainieren ist zeitverschwendung
            print("learning rate to low, stop training")
            break
        if steps_without_improvment > 10:
            print("no imprevment of val_loss, stop training")
            break
    dt = time.time()-start_time
    print(saveName, " took ", dt, "s to fit")

    #start_time = time.time()
    #model.fit(x_train, y_train, epochs=64, steps_per_epoch=len(x_train), verbose=0)
    #dt = time.time()-start_time
    #print(saveName, " took ", dt, "s to fit\n")
    dir = "C:\\Users\\Idefix\\PycharmProjects\\data\\"
    # dir = Dataloader.data_dir+"/models/"
    model.save(dir+saveName+".h5")
    return history


""" 
kommentar:
    with open(output_path + "/history.txt", 'w') as f:
        for k in ae.history.history.keys():
            print(k, file=f)
            for i in ae.history.history[k]:
                print(i, file=f)
                
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


if __name__ == "__main__":
    Dataloader.test_tfds()
    exit(0)
    #data = Dataloader.getData(dir=Dataloader.data_dir, img_type=Dataloader.img_types.paragraph, goldlabel_type=Dataloader.goldlabel_types.linepositions, goldlabel_encoding=Dataloader.goldlabel_encodings.dense, maxcount=200)
    train_data, val, test = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.onehot)
    x_shape = train_data[0][0].shape
    model = Models.getModel("findfollowreadlite_mse", (x_shape[0], x_shape[1], 1), train_data[1][0].shape[0])
    train(model, "sparse_mse", train_data[0], train_data[1], val)


def test_test():
    """
    uses the simplier find_the_centor_of_the_point dataset and all models
    """
    # test model
    tc_dense_mse = tf.keras.models.load_model("C:/Users/Idefix/PycharmProjects/data/"+"tc_dense_mse"+".h5")
    tc_dense_cce = tf.keras.models.load_model("C:/Users/Idefix/PycharmProjects/data/"+"tc_dense_cce"+".h5")
    tc_sparse_mse = tf.keras.models.load_model("C:/Users/Idefix/PycharmProjects/data/"+"tc_sparse_mse"+".h5")
    tc_sparse_cce = tf.keras.models.load_model("C:/Users/Idefix/PycharmProjects/data/"+"tc_sparse_cce"+".h5")
    imgs_dense = Dataloader.get_testdata(enc=Dataloader.point2dense)
    imgs_sparse = Dataloader.get_testdata(enc=Dataloader.point2spares)
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
    exit(0)

    # train models on test circle data
    imgs_d = Dataloader.get_testdata(enc=Dataloader.point2dense)
    x = np.array([d[0] for d in imgs_d])
    y = np.array([np.array(d[1], dtype=float) for d in imgs_d])
    y_size = y[0].shape[0]
    modle = Models.getModel("findfollowreadlite_mse", (32, 32, 1), y_size)
    hist = train(modle, "tc_dense_mse", x, y, val=None)
    print("history of tc_dense_mse: \n   loss: ", hist['loss'], "\n   vall: ", hist['val_loss'], "\n   lr__: ", hist['lr'])
    modle = Models.getModel("findfollowreadlite_cce", (32, 32, 1), y_size)
    train(modle, "tc_dense_cce", x, y, val=None)

    imgs_s = Dataloader.get_testdata(enc=Dataloader.point2spares)
    x = np.array([d[0] for d in imgs_s])
    y = np.array([np.array(d[1], dtype=float) for d in imgs_s])
    y_size = y[0].shape[0]
    modle = Models.getModel("findfollowreadlite_mse", (32, 32, 1), y_size)
    hist = train(modle, "tc_sparse_mse", x, y, val=None)
    print("history of tc_sparse_mse: \n   loss: ", hist['loss'], "\n   vall: ", hist['val_loss'], "\n   lr__: ", hist['lr'])
    modle = Models.getModel("findfollowreadlite_cce", (32, 32, 1), y_size)
    hist = train(modle, "tc_sparse_cce", x, y, val=None)
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
    infer("dense_mse", decode_func=Dataloader.dense2linepoints)


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

#original training
#error of  dense_mse  =  1366, 180s
#error of  dense_cce  =  21803, 180s
#error of  sparse_mse  =  869, 200s
#error of  sparse_cce  =  31744, 200s

#training with adaptive lr
# dense_mse 309s, 1041
# dense_cce  356s, 21823
# sparse_mse 674s, 393
# sparse_cce 511s, 31744

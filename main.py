import cv2
import numpy as np
import tensorflow as tf
import time

import Dataloader
import Models
import keras.losses

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
            cv2.circle(img, point[0], int(point[2]), 125, 2)
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
    print("len(x_train) = ", len(x_train))
    print("train: ", x_train[0], " -> ", y_train[0])
    print("x.shape = ", x_train[0].shape)
    print("y.shape = ", y_train[0].shape)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=2, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    start_time = time.time()
    model.fit(x_train, y_train, epochs=128, steps_per_epoch=len(x_train), callbacks=[callback], validation_data=val)
    dt = time.time()-start_time
    print(saveName, " took ", dt, "s to fit")
    model.save("../SimpleHTR/data/modles/"+saveName+".h5")


def infer(name, decode_func=Dataloader.sparse2points):
    (x, y), (val_x, val_y), (test_x, test_y) = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.dense)
    model = tf.keras.models.load_model(name)
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

    # test_x = [np.pad(img, ((0, input_size[0]-img.shape[0]), (0, input_size[1]-img.shape[1])), mode='constant', constant_values=255) for img in test_x]
    # print("test_x: ", type(test_x[0]))
    # print("test_y: ", type(test_y[0]))
    # loss, acc = model.evaluate(test_x, test_y, verbose=2)
    # print("model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == "__main__":
    #data_dir = "C:\\Users\\Idefix\\PycharmProjects\\SimpleHTR\\trainingDataset"
    #data_dense = (x_train, y_train), (x_val, y_val), (x_test, y_test)
    train_data, val, test = Dataloader.getTrainingData(Dataloader.goldlabel_encodings.dense)  # getTrainingData() = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    mse_loss = keras.losses.MeanSquaredError()
    cce_loss = keras.losses.CategoricalCrossentropy()
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

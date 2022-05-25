# fullpageHTR
Train NN to read/detect text in lines/paragraph images.<br>
Based on Tensorflow <br>

## structure
Dataloader contains the function/classes to load IAM word images and assemble them to line/paragraph imgages with text/lineposition/lineimg goldlabels. <br>
Main contains funtions to train models and inspect trained models. <br>
Models contains all tensorflow models. <br>

### Dataloader
#### Datatypes
line_point is a datastructures to describe the position and size of straight textlines in an image.
#### Dataset
To not be confused with tf.data.dataset. They have no relation at all. <br>
Holds metainformation of what type of data should be generated.
More dokumentation in Dataloader.Dataset and Dataloader.abstractDataset

### Models
#### nameing
model names have the format f"{ds.name}_{model.name}_relu_hard_sigmoid_t1tfds{maxdata}_{batch_size}".
where ds is a Dataloader.Dataset and model is a tensorflow model. maxdata and batch-size are parameters for the train method.

## Working Models
Dataset_real(22_(128, 256), 1_15)_conv_relu_hard_sigmoid_t1tfds100000_32  took  186249.77829027176 s to fit to val_loss of  0.024048075079917908
Dataset_real(22_(128, 256), 1_15)_conv2_relu_hard_sigmoid_t1tfds100000_32  took  9632.773826122284 s to fit to val_loss of  0.01800730638206005

Despite the higher val_loss, the results of conv seem to be more accurate than the results of conv2.
But both models struggle to predict the end of long lines.


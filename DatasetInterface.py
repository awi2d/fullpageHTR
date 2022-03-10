import tensorflow as tf
import numpy as np

# Write the records to a file.

def writeData(filePath, data):
    """
    no clue how this works, just copied from https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    :param filePath:
    path to directory where data should be stored
    :param data:
    type [(input, goldlabel)]
    e.g. [(image_of_handwritten_paragraph, positions of lines on that image)]
    """
    with tf.io.TFRecordWriter(filePath) as file_writer:
        for (x, y) in data:

            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
                "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
            })).SerializeToString()
            file_writer.write(record_bytes)


def readData(filePath):
    return [("x1", "y1"), ("x2", "y2")]


def decode_fn(record_bytes):
    # Read the data back out.
    return tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"x": tf.io.FixedLenFeature([], dtype=tf.float32),
         "y": tf.io.FixedLenFeature([], dtype=tf.float32)}
    )

import tensorflow as tf
import joblib


data = joblib.load("/Volumes/Transcend/repositories/Aglaia/data/local_slatm_qm7_light.bz")
descriptors = data['descriptor']
zs = data["zs"]
energies = data["energies"]

dataset = tf.data.Dataset.from_tensor_slices((descriptors, zs, energies))
dataset_shuff = dataset.shuffle(buffer_size=100)
batched_dataset = dataset_shuff.batch(20)
iterator = tf.data.Iterator.from_structure(batched_dataset.output_types, batched_dataset.output_shapes)
x_tf, zs_tf, y_tf = iterator.get_next()

training_init_op = iterator.make_initializer(batched_dataset)

with tf.Session() as sess:
    sess.run(training_init_op)

    while True:
        try:
            elem = sess.run(y_tf)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
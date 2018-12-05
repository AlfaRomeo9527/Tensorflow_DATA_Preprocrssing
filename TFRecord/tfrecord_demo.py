import tensorflow as tf
import PIL.Image as Image
import numpy as np

'''
# 生成TFRecods
tfrecords_filename = './train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# 基本的，一个Example中包含Features，Features里包含Feature（这里没s）的字典。最后，Feature里包含有一个 FloatList， 或者ByteList，或者Int64List

LABEL = 3
image = Image.open('./anonym.jpg')
height, width = image.size
image_byte_data = np.array(image).tostring()
print(image_byte_data)
example = tf.train.Example(
    features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[LABEL])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_byte_data]))
    }))
writer.write(example.SerializeToString())  # 序列化为字符串
writer.close()


#值得注意的是赋值给example的数据格式。从前面tf.train.Example的定义可知，tfrecord支持整型、浮点数和二进制三种格式，分别是:

tf.train.Feature(int64_list = tf.train.Int64List(value=[int_scalar]))
tf.train.Feature(bytes_list = tf.train.BytesList(value=[array_string_or_byte]))
tf.train.Feature(bytes_list = tf.train.FloatList(value=[float_scalar]))


for serialized_example in tf.python_io.tf_record_iterator("./train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image_data'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    height = example.features.feature['height'].int64_list.value
    width = example.features.feature['width'].int64_list.value
    print(label)
    print(width)
    print(height)
'''

#
# tf.train.Feature(int64_list = tf.train.Int64List(value=[int_scalar]))
# tf.train.Feature(bytes_list = tf.train.BytesList(value=[array_string_or_byte]))
# tf.train.Feature(bytes_list = tf.train.FloatList(value=[float_scalar]))



import tensorflow as tf
import numpy as np
import os


# =============================================================================#
# write images and label in tfrecord file and read them out
def encode_to_tfrecords(tfrecords_filename, data_num):
    # write into tfrecord file
    if os.path.exists(tfrecords_filename):
        os.remove(tfrecords_filename)

    writer = tf.python_io.TFRecordWriter('./' + tfrecords_filename)  # 创建.tfrecord文件，准备写入

    for i in range(data_num):
        img_raw = np.random.randint(0, 255, size=(56, 56))
        img_raw = img_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())

    writer.close()
    return 0


def decode_from_tfrecords(filename, is_batch):
    # 根据文件名生成队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.int64)
    image = tf.reshape(image, [56, 56])
    label = tf.cast(features['label'], tf.int64)

    if is_batch:
        batch_size = 5
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        image, label = tf.train.shuffle_batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return image, label


if __name__ == "__main__":
    encode_to_tfrecords('eval.tfrecord', 5)
    image, lable = decode_from_tfrecords('/home/alpha/MyProject/Tensorflow_DATA_Preprocrssing/TFRecord/eval.tfrecord',True)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()#创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)#启动QueueRunner, 此时文件名队列已经进队。
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                print((sess.run(image)).shape)
                sess.run(lable)

        except tf.errors.OutOfRangeError:
            print
            'Done training -- epoch limit reached'
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)



'''
tf.train.shuffle_batch函数输入参数为：

tensor_list: 进入队列的张量列表The list of tensors to enqueue.
batch_size: 从数据队列中抽取一个批次所包含的数据条数The new batch size pulled from the queue.
capacity: 队列中最大的数据条数An integer. The maximum number of elements in the queue.
min_after_dequeue: 提出队列后，队列中剩余的最小数据条数Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
num_threads: 进行队列操作的线程数目The number of threads enqueuing tensor_list.
seed: 队列中进行随机排列的随机数发生器，似乎不常用到Seed for the random shuffling within the queue.
enqueue_many: 张量列表中的每个张量是否是一个单独的例子，似乎不常用到Whether each tensor in tensor_list is a single example.
shapes: (Optional) The shapes for each example. Defaults to the inferred shapes for tensor_list.
name: (Optional) A name for the operations.
值得注意的是，capacity>=min_after_dequeue+num_threads*batch_size。
'''
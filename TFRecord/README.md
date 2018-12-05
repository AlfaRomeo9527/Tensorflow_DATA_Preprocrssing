###TFRecord

#### 概述：

关于Tensorflow读取数据，官网给出了三种方法：
1. 供给数据：在tensorflow程序运行的每一步，让python代码供给数据，即每一个Batch feed_dict{}一次。
2. 从文件读取数据：建立输入pipline，从文件中读取。
3. 预加载数据：如果数据量不大，可以在程序中定义常量或者变量来保存所有的数据。  

***

本文主要介绍第二种：TFRecord是TensorFlow官方推荐使用的数据格式化存储工具，它不仅规范了数据的读写方式，还大大地提高了IO效率。  

---

#### TFRecords  
对于数据量较小而言，可能一般选择直接将数据加载进内存，然后再分batch输入网络进行训练（tip:使用这种方法时，结合yield 使用更为简洁，大家自己尝试一下吧，我就不赘述了）。但是，如果数据量较大，这样的方法就不适用了，因为太耗内存，所以这时最好使用tensorflow提供的队列queue，也就是第二种方法 从文件读取数据。对于一些特定的读取，比如csv文件格式，官网有相关的描述，在这儿我介绍一种比较通用，高效的读取方法（官网介绍的少），即使用tensorflow内定标准格式——TFRecords
TFRecords文件包含了tf.train.Example 协议内存块(protocol buffer)(协议内存块包含了字段 Features)。我们可以写一段代码获取你的数据， 将数据填入到Example协议内存块(protocol buffer)，将协议内存块序列化为一个字符串， 并且通过tf.python_io.TFRecordWriter 写入到TFRecords文件。

从TFRecords文件中读取数据， 可以使用tf.TFRecordReader的tf.parse_single_example解析器。这个操作可以将Example协议内存块(protocol buffer)解析为张量。  
tensorflow/g3doc/how_tos/reading_data/convert_to_records.py就是这样的一个例子。  
tf.train.Example的定义如下：    
 
    message Example { 
        Features features = 1;
    };  
    message Features{
      map<string,Feature> featrue = 1;
    };  
    message Feature{    
        oneof kind{        
                BytesList bytes_list = 1;        
                FloatList float_list = 2;        
                Int64List int64_list = 3;    
        }
    };  

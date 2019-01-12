## Building Image Classifier with Tensorflow
We are building an Image classifier in Tensorflow with custom input pipeline.
In this project we are trying to classify whether the given image contains cat or dog.
The dataset is downloaded from kaggle.

### Requirements:
```
1. Numpy
2. Tensorflow
3. OpenCV
```

### Steps
```
1. Import libraries
2. Import dataset
3. Create Model
4. Train and Test Model
```

#### Step 1: Import Libraries
```
import os
import numpy as np
import tensorflow as tf
import cv2
from random import shuffle
import pandas as pd
```

#### Step 2: Creating input pipeline
```
def generate_data():
  train_data = []
  for file in os.listdir('data/train'):
    img = cv2.imread(os.path.join('data/train',file),0)
    img = cv2.resize(img,(56,56))
    if(file[0:3]=='cat'):
      train_data.append([img,'0'])
    else:
      train_data.append([img,'1'])
   shuffle(train_data)   
   return train_data     
```
- First we will create an empty list.
- We will list filenames in the train directory using 'os.listdir'.
- Using 'cv2.imread', we will read images(0 paramter reads image in grayscale).
- We will resize the image according to the need(These images will stored in RAM. Choose image size according to dataset size and RAM     capacity).
- If starting 3 letters of the image are 'cat', we will add 0 as label and if it is 'dog' then 1.
- Return train_data.

#### Step 3: Generating training data
```
train_data = generate_data()
train_x = [train[i][0] for i in range(len(train_data))]
train_y = [train[i][1] for i in range(len(train_data))]
```
- We are creating training data using 'generate_data()' function.
- we are using list comprehension to collect images and labels associated with them
- train_data[x][y] -> x is row and y is column.
- train[i][0] -> it means we are accessing rows of 0 column.
- train[i][1] -> it means we are accessing rows of 1 column.

Next step is converting them into numpy array
```
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
```

#### Step 4: Resizing the data:
```
train_x.shape, train_y.shape
```
((25000, 56, 56), (25000,))
- Train data should be a 4D tensor.
- So, we will resize it.
```
train_x = train_x.resize(-1,56,56,1)
```
- The first parameter -1 tells that it can anything. Therefore its value is calculated automatically.

Now, we will use 'pd.get_dummies()' function to create create target vector.
```
train_y = pd.get_dummies(train_y)
```
```
train_x.shape, train_y.shape
```
((25000, 56, 56, 1), (25000,2))

#### Step 5: Initializing variables and placeholders:
```
epoch = 50
learning_rate = 0.01
batch_size = 64
n_classes = 2
```

**Placeholder**: Something which we use to feed input.
```
x = tf.placeholder(tf.float32,[None,56,56,1]) # None will be automatically replaced by batch_size
y = tf.placeholder(tf.float32,[None,n_classes]) # None will be replaced by number of neurons in the hidden layer.
```
None paramter tells that it can be anything.

#### Step 6: Creating CNN:
```
def conv2d(x,w,b,stride=1):
  x = tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding='SAME')
  x = tf.nn.bias_add(x,b)
  return tf.nn.relu(x)

def max_pool(x,stride=2):
  return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding='SAME')
```  

**Initializing weights and biases**
```
weight={
    'wc1':tf.get_variable('w0',shape=(3,3,1,32),initializer=tf.contrib.layers.xavier_initializer()),
    'wc2':tf.get_variable('w1',shape=(3,3,32,64),initializer=tf.contrib.layers.xavier_initializer()),
    'wc3':tf.get_variable('w2',shape=(3,3,64,64),initializer=tf.contrib.layers.xavier_initializer()),
    'wc4':tf.get_variable('w3',shape=(3,3,64,128),initializer=tf.contrib.layers.xavier_initializer()),
    'wd0':tf.get_variable('wde0',shape=(4*4*128,128),initializer=tf.contrib.layers.xavier_initializer()),
    'wd1':tf.get_variable('wde1',shape=(128,2),initializer=tf.contrib.layers.xavier_initializer())
}

biases={
    'bc1':tf.get_variable('b0',shape=(32),initializer=tf.contrib.layers.xavier_initializer()),
    'bc2':tf.get_variable('b1',shape=(64),initializer=tf.contrib.layers.xavier_initializer()),
    'bc3':tf.get_variable('b2',shape=(64),initializer=tf.contrib.layers.xavier_initializer()),
    'bc4':tf.get_variable('b3',shape=(128),initializer=tf.contrib.layers.xavier_initializer()),
    'bd0':tf.get_variable('bde0',shape=(128),initializer=tf.contrib.layers.xavier_initializer()),
    'bd1':tf.get_variable('bde1',shape=(2),initializer=tf.contrib.layers.xavier_initializer())
}
```
shape = (3,3,1,32)
- 3,3 is the filter size.
- 1 is the depth(grayscale), for coloured it will be 3.
- 32 means the number of filters.

Now, we will create the model
```
def conv_net(x,weight,biases):
    conv1 = conv2d(x,weight['wc1'],biases['bc1'])
    conv1 = max_pool(conv1,2)
    
    conv2 = conv2d(conv1,weight['wc2'],biases['bc2'])
    conv2 = max_pool(conv2,2)
    
    conv3 = conv2d(conv2,weight['wc3'],biases['bc3'])
    conv3 = max_pool(conv3,2)
    
    conv4 = conv2d(conv3,weight['wc4'],biases['bc4'])
    conv4 = max_pool(conv4,2)
    
    fc1 = tf.reshape(conv4, [-1, weight['wd0'].get_shape().as_list()[0]])
    
    fc1 = tf.add(tf.matmul(fc1, weight['wd0']), biases['bd0'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weight['wd1']), biases['bd1'])
    
    return out
```
Creating cost function and optimization function
```
pred = conv_net(x,weight,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
```
```
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```
```
init = tf.global_variables_initializer()
```
```
```
with tf.Session() as sess:
    sess.run(init)
    
    training_loss = []
    test_loss = []
    training_accuracy = []
    test_accuracy = []
    
    for _ in range(epochs):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
            
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
            loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y})
            
            print("Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
  ```      
       


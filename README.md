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








**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)



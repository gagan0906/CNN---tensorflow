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
    img = cv2.resize(img,(128,128)
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

#### Step 4: Resizing the data
```
train_x.shape, train_y.shape
```



1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/gagan0906/portfolio/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

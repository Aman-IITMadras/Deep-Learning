import cv2                 # mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # Not to loss patience as images load

TRAIN_DIR = 'E:/FashionDx/Training Set'  # Enter File Path of your traning folder, with sub-folders of "Training Set", as provided in Drive, "Checks","Solids","Stripes"
TEST_DIR = 'E:/FashionDx/Test'           # Enter File Path of testing Data.(12 Labeled images Visible in output) (12 images of various kinds,"Solids","Stripes","Checks")
IMG_SIZE = 400                           # Enter Image Size
LR = 1e-4                                # Enter Learning Rate, Now 10^-4

MODEL_NAME = 'FashionDX-{}-{}.model'.format(LR, '2conv-basic')

def create_train_data():
    training_data = []

    TRAIN_DIR_CHECKS = os.path.join(TRAIN_DIR,'checks')   # Training Data in Checks folder

    TRAIN_DIR_CHECKS_LARGE = os.path.join(TRAIN_DIR_CHECKS,'Large')
    for img in tqdm(os.listdir(TRAIN_DIR_CHECKS_LARGE)): # Run through traing data in "checks/large" folder
        label = [1,0,0,0,0,0,0]                          # one hot array [checks_large, checks_medium,checks_small,solids,stripes_dense,stripes_large, stripes_medium]
        path = os.path.join(TRAIN_DIR_CHECKS_LARGE,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)]) # array of images and respective one hot arrays

    TRAIN_DIR_CHECKS_MED = os.path.join(TRAIN_DIR_CHECKS,'Medium')
    for img in tqdm(os.listdir(TRAIN_DIR_CHECKS_MED)):
        label = [0,1,0,0,0,0,0]
        path = os.path.join(TRAIN_DIR_CHECKS_MED,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    TRAIN_DIR_CHECKS_SMALL = os.path.join(TRAIN_DIR_CHECKS,'Small')
    for img in tqdm(os.listdir(TRAIN_DIR_CHECKS_SMALL)):
        label = [0,0,1,0,0,0,0]
        path = os.path.join(TRAIN_DIR_CHECKS_SMALL,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    TRAIN_DIR_SOLIDS = os.path.join(TRAIN_DIR,'solids') # Training Data in Solids folder
    for i in range(3):
     for img in tqdm(os.listdir(TRAIN_DIR_SOLIDS)):
         label = [0,0,0,1,0,0,0]
         path = os.path.join(TRAIN_DIR_SOLIDS,img)
         img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
         img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
         training_data.append([np.array(img),np.array(label)])

    TRAIN_DIR_STRIPES = os.path.join(TRAIN_DIR,'stripes') #Training Data in stripes folder

    TRAIN_DIR_STRIPES_D = os.path.join(TRAIN_DIR_STRIPES,'vertical_dense_stripes')
    for img in tqdm(os.listdir(TRAIN_DIR_STRIPES_D)):
        label = [0,0,0,0,1,0,0]
        path = os.path.join(TRAIN_DIR_STRIPES_D,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    TRAIN_DIR_STRIPES_L = os.path.join(TRAIN_DIR_STRIPES,'vertical_large_stripes')
    for img in tqdm(os.listdir(TRAIN_DIR_STRIPES_L)):
        label = [0,0,0,0,0,1,0]
        path = os.path.join(TRAIN_DIR_STRIPES_L,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    TRAIN_DIR_STRIPES_M = os.path.join(TRAIN_DIR_STRIPES,'vertical_medium_stripes')
    for img in tqdm(os.listdir(TRAIN_DIR_STRIPES_M)):
        label = [0,0,0,0,0,0,1]
        path = os.path.join(TRAIN_DIR_STRIPES_M,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)   # Save traning data
    return training_data

def process_test_data():
    testing_data = []
    j = 1
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = j
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        j+=1

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
#If you have already created the dataset:
#train_data = np.load('train_data.npy')     # Use if already created training data befor

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')


convnet = conv_2d(convnet, 32, 5, activation='relu')    # Used ReLU activation function
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)


convnet = fully_connected(convnet, 7, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# Activation: Softmax
# Optimizer: Adam

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):     # Load Saved Model
    #model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-100]                           # Use ALL but last 100 for training
test = train_data[-100:]                            # Use Last 100 images to check accuracy, loss

X= np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=18, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME) # Model is giving Accurate Results with New Data,(12 images in Output), OVERFITTING hasn't occured

#model.save(MODEL_NAME)                                      # Save Model

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score  # Display Precision,Recall,Fscore,confusion_matrix
y_true = []
y_pred = []
for num,data in enumerate(train_data[-100:]):

    y_true.append(np.argmax(data[1]))

    img_data = data[0]
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    y_pred.append(np.argmax(model_out))

print("Precision", precision_score(y_true,y_pred,average='macro'))
print("Recall", recall_score(y_true, y_pred,average='macro'))
print("f1_score", f1_score(y_true, y_pred,average='macro'))
print("confusion_matrix", confusion_matrix(y_true,y_pred))

import matplotlib.pyplot as plt                             # Visually inspect our network against unlabeled data, Create a output window of 12 Test images

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')                       # Use if already created test data

fig=plt.figure()

for num,data in enumerate(test_data[:]):

    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,num+1)                         # 12 test images displayed in 3x4, with labels
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 3:                         # Label 12 Test images as "Solids", "Checks", "Stripes"
        str_label = 'Solids'
    if np.argmax(model_out) == 0 :
        str_label='Checks'
    if np.argmax(model_out) == 1:
        str_label='Checks'
    if np.argmax(model_out) == 2:
        str_label='Checks'
    if np.argmax(model_out) == 4 :
        str_label= 'Stripes'
    if np.argmax(model_out) == 5 :
        str_label= 'Stripes'
    if np.argmax(model_out) == 6 :
        str_label= 'Stripes'

    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)              # Hide x,y axis
    y.axes.get_yaxis().set_visible(False)
plt.show()

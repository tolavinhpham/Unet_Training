import tensorflow as tf
import os
import random
import numpy as np
import keras as kr

from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = 'C:/Users/ASUS/OneDrive - Hanoi University of Science and Technology/Research/train_set'
TEST_PATH = 'C:/Users/ASUS/OneDrive - Hanoi University of Science and Technology/Research/test_set'
layer_names = ['input_1', 'lambda', 'conv2d', 'dropout', 'conv2d_1', 'max_pooling2d', 'conv2d_2', 'dropout_1', 'conv2d_3', 'max_pooling2d_1', 'conv2d_4', 'dropout_2', 'conv2d_5', 'max_pooling2d_2', 'conv2d_6', 'dropout_3', 'conv2d_7', 'max_pooling2d_3', 'conv2d_8', 'dropout_4', 'conv2d_9', 'conv2d_transpose', 'concatenate', 'conv2d_10', 'dropout_5', 'conv2d_11', 'conv2d_transpose_1', 'concatenate_1', 'conv2d_12', 'dropout_6', 'conv2d_13', 'conv2d_transpose_2', 'concatenate_2', 'conv2d_14', 'dropout_7', 'conv2d_15', 'conv2d_transpose_3', 'concatenate_3', 'conv2d_16', 'dropout_8', 'conv2d_17', 'conv2d_18']


train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH
    img = imread(path + "/" + id_ + "/images/" + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
    
    # Add images each others
    for mask_file in next(os.walk(path + "/" + id_ + '/masks/'))[2]:
        mask_ = imread(path + "/" + id_ + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train[n] = mask   

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Check the data all right
# image_x = random.randint(0, len(train_ids))
# imshow(X_train[image_x])
# plt.show()
# imshow(np.squeeze(Y_train[image_x]))
# plt.show()


print("Shape of X_test:", X_test.shape)
# Print the shape of a sample training image (assuming you have access to one)


#Build the model
inputs = kr.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = kr.layers.Lambda(lambda x: x / 255)(inputs) # Convert uint8 to float

#Contraction path
c1 = kr.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = kr.layers.Dropout(0.1)(c1)
c1 = kr.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = kr.layers.MaxPooling2D((2, 2))(c1)

c2 = kr.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = kr.layers.Dropout(0.1)(c2)
c2 = kr.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = kr.layers.MaxPooling2D((2, 2))(c2)
 
c3 = kr.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = kr.layers.Dropout(0.2)(c3)
c3 = kr.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = kr.layers.MaxPooling2D((2, 2))(c3)
 
c4 = kr.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = kr.layers.Dropout(0.2)(c4)
c4 = kr.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = kr.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = kr.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = kr.layers.Dropout(0.3)(c5)
c5 = kr.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = kr.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = kr.layers.concatenate([u6, c4])
c6 = kr.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = kr.layers.Dropout(0.2)(c6)
c6 = kr.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = kr.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = kr.layers.concatenate([u7, c3])
c7 = kr.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = kr.layers.Dropout(0.2)(c7)
c7 = kr.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = kr.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = kr.layers.concatenate([u8, c2])
c8 = kr.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = kr.layers.Dropout(0.1)(c8)
c8 = kr.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = kr.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = kr.layers.concatenate([u9, c1], axis=3)
c9 = kr.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = kr.layers.Dropout(0.1)(c9)
c9 = kr.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = kr.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = kr.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
model.summary()

################################
# Modelcheckpoint (Save model after every epoch)
# Save the best only
checkpointer = kr.callbacks.ModelCheckpoint('model_for_nuclei.keras', verbose=1, save_best_only=True)

# To make sure the stop point is the best point
callbacks = [
        kr.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        kr.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

####################################

# Predict in random image
idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions to binarize the image
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()


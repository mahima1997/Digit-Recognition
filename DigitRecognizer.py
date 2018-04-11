import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


train = np.loadtxt("train.csv", skiprows=1, dtype='int', delimiter=',')   #outputs an array
test= np.loadtxt("test.csv",skiprows=1, dtype='int', delimiter=',')


X_train, X_val, y_train, y_val = train_test_split(train[:,1:], train[:,0], test_size=0.2, random_state=1234)

#'Input data in `NumpyArrayIterator` should have rank 4. So,
x_train = X_train.reshape(-1, 28, 28, 1)
x_val = X_val.reshape(-1, 28, 28, 1)

x_train = x_train.astype("float32")/255.
x_val = x_val.astype("float32")/255.
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  #as there are 10 category of images

#DATA AUGMENTATION
datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])

#We then speed things up, only to reduce the learning rate by 10% every epoch.
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=50, 
                           verbose=2,  
                           validation_data=(x_val[:400,:], y_val[:400,:]),
                           callbacks=[annealer])

final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

model.save_weights("digit_recog_model_weights.h5")
model.save('digit_recog_model.h5')

y_hat = model.predict(x_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)

#SUBMISSION

test=test.reshape(-1, 28, 28, 1)
test = test.astype("float32")/255

y_hat = model.predict(test)
y_pred = np.argmax(y_hat,axis=1)

with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(y_pred)) :
        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))

print("Done")



import csv
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

csvfile = './examples/driving_log.csv'
imgfolder = './examples/IMG/'

# Read driving_log.csv into an array of lines of text
# Each line of lines is an array with format
# ['Path center camera', 'Path left camera', 'Path right camera', Steering angle, Throttle, Brake, Speed]

def loaddata (csvfile):
    lines = []
    with open(csvfile) as input:
        reader = csv.reader(input)
        for line in reader:
            lines.append(line)
    return lines[1:]


def preparedata (samples):
    # On demand loading of data depending on batch sizes
    images = []
    angles = []
    for sample in samples:
        # Extract filenames (stripped of directory path) for 
        # this sample's center, left, and right images
        filename_center = (sample[0].split('\\')[-1]).strip()
        filename_left = (sample[1].split('\\')[-1]).strip()
        filename_right = (sample[2].split('\\')[-1]).strip()
        # Construct image paths relative to model.py 
        path_center = imgfolder + filename_center
        path_left = imgfolder + filename_left
        path_right = imgfolder + filename_right

        image_center = mpimg.imread(path_center)
        image_left = mpimg.imread(path_left)
        image_right = mpimg.imread(path_right)
        # In addition to the center, left, and right camera images,
        # we augment with a left-right flipped version of the center, keft and right camera's image.
        image_center_flipped = np.copy(np.fliplr(image_center))
        image_right_flipped = np.copy(np.fliplr(image_right))
        image_left_flipped = np.copy(np.fliplr(image_left))

        images.append(image_center)
        images.append(image_left)
        images.append(image_right)
        images.append(image_center_flipped)
        images.append(image_right_flipped)
        images.append(image_left_flipped)

        correction = 0.085
        angle_center = float(sample[3])
        angle_left = angle_center + correction
        angle_right = angle_center - correction
        angle_center_flipped = -angle_center
        angle_right_flipped = -angle_right
        angle_left_flipped = -angle_left

        angles.append(angle_center)
        angles.append(angle_left) 
        angles.append(angle_right)
        angles.append(angle_center_flipped)
        angles.append(angle_right_flipped)
        angles.append(angle_left_flipped)
    return images, angles
    
# Generator for fit data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever, so that next() can be called on the generator 
    # indefinitely over  arbitrarily many epochs.
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]     
            images, angles = preparedata(batch_samples)
            # Return a training batch of size 4*batch_size to model.fit_generator
            X_train = np.array(images)
            y_train = np.array(angles) 
            yield sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(loaddata(csvfile), test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

# Define generators for training and validation data, to be used with fit_generator below
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Crop the hood of the car and the higher parts of the images 
model.add( Cropping2D( cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#Normalize the data.
model.add( Lambda(lambda x: x/255. - 0.5))
# Nvidia Network
# Convolution Layers
model.add( Convolution2D(24, 5, 5, subsample=(2,2), activation = 'relu'))
model.add( Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add( Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add( Convolution2D(64, 3, 3, subsample=(1,1), activation = 'relu'))
model.add( Convolution2D(64, 3, 3, subsample=(1,1), activation = 'relu'))
# Flatten for transition to fully connected layers.
model.add(Flatten())
# Fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Use mean squared error for regression, and an Adams optimizer.
model.compile(loss='mse', optimizer='adam')

train_steps = np.ceil(len(train_samples)/32).astype(np.int32)
validation_steps = np.ceil(len( validation_samples)/32).astype(np.int32)

#model.fit( X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose = 1)
model.fit_generator(train_generator, \
          steps_per_epoch = train_steps, \
          epochs=15, \
          verbose=1, \
          callbacks=None, 
          validation_data=validation_generator, \
          validation_steps=validation_steps, \
          class_weight=None, \
          max_q_size=10, \
          workers=1, \
          pickle_safe=False, \
          initial_epoch=0)

model.save('model.h5')

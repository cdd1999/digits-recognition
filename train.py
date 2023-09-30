from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=6, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


from keras.preprocessing.image import ImageDataGenerator

image_data_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_image_data_gen = ImageDataGenerator(rescale=1./255)

training_set = image_data_gen.flow_from_directory('data/trainSet',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')
test_set = test_image_data_gen.flow_from_directory('data/testSet',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
classifier.fit(
        training_set,
        steps_per_epoch=len(training_set), # 600 images
        epochs=20,
        validation_data=test_set,
        validation_steps=len(test_set)) # No of images in test set

classifier.save("model.h5")
# Saving the model
model_json = classifier.to_json()
with open("model-weights.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-weights.h5')


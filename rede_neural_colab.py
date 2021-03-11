# -*- coding: utf-8 -*-

"""
**Importanto as bibliotecas**
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import numpy as np
tf.__version__

"""**Carregando a base de dados**"""

from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive/covid_dataset.zip"
zip_object = zipfile.ZipFile(file=path, mode="r")
zip_object.extractall("./")
zip_object.close()

"""**Carregando uma imagem de pessoa com convid-19**"""

image = tf.keras.preprocessing.image.load_img(r'/content/covid_dataset/train/covid/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg', target_size=(224,224))
plt.imshow(image);

"""**Carregando uma imagem sem convid 19**"""

image = tf.keras.preprocessing.image.load_img(r'/content/covid_dataset/train/normal/NORMAL2-IM-1281-0001.jpeg', target_size=(224, 224))
plt.imshow(image);

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                   rotation_range = 50,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   zoom_range = 0.1,
                                   horizontal_flip = True,
                                   vertical_flip = True)

train_generator = train_datagen.flow_from_directory('/content/covid_dataset/train',
                                                    target_size = (224, 224),
                                                    batch_size=16,
                                                    class_mode = 'categorical',
                                                    shuffle = True)


step_size_train = train_generator.n // train_generator.batch_size

test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_generator = test_datagen.flow_from_directory('/content/covid_dataset/test',
                                                  target_size=(224,244),
                                                  batch_size=1,
                                                  class_mode = 'categorical',
                                                  shuffle = False)

step_size_test = test_generator.n // test_generator.batch_size

"""**Aplicando técnica de transferência de aprendizagem**"""

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

base_model.summary()

x = base_model.output

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
preds = tf.keras.layers.Dense(2, activation='softmax')(x)

model = tf.keras.Model(inputs = base_model.input, outputs = preds)

model.summary()

for i, layer in enumerate(model.layers):
  print(i, layer.name)

for layer in model.layers[:175]:
  layer.trainable = False

for layer in model.layers[175:]:
  layer.trabainle = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              epochs=200,
                              steps_per_epoch=step_size_train,
                              validation_data = test_generator,
                              validation_steps=step_size_test)

"""**Avaliação da Rede Neural**"""

# Acurácia Média
np.mean(history.history['val_accuracy'])

# Desvio padrão
np.std(history.history['val_accuracy'])

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend();

plt.axis([0, 200, 0, 1])
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend();

filenames = test_generator.filenames
len(filenames)

predictions = model.predict_generator(test_generator, steps = len(filenames))

print(predictions)

predictions2 = []
for i in range(len(predictions)):
  #print(predictions[i])
  predictions2.append(np.argmax(predictions[i]))

print(predictions2)

print(test_generator.classes)

print(test_generator.class_indices)

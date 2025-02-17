
#####################################   Deep Nueral Network    ################################################

# pip install tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt

# tensorflow open source library for complex numerical computation
# keras high level API Application Programming Interface for building and trainingg deep learning models
#  functions keras
mist = tf.keras.datasets.mnist # 70000 image size 28*28

(x_train, y_train),(x_test, y_test) = mist.load_data()
# print(len(x_train))
# print(len(x_test))



fig, axes = plt.subplots(2,5, figsize=(15,6))
# axes <-- 2D
# flatten converts axes into 1D [0,1,2,3,4,5,6,7,8,9]
for idx, axe in enumerate(axes.flatten()):
    axe.axis('off')
    axe.set_title(f'Label : {y_train[idx]}')
    axe.imshow(x_train[idx])
plt.show()

#normalization by converting 0-255 to [0-1]
x_train,x_test = x_train/255.0, x_test/255.0

# print(x_train[0])
# exit()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), # 784 
    tf.keras.layers.Dense(128, activation ='relu'), # First hidden Layer Dense acceptes only FC fully connected layer means 1D
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation ='relu'), # Second hidden layer
    tf.keras.layers.Dense(10, activation='softmax' )
])

#'adam The Adam optimizer helps the model learn by adjusting its internal numbers (weights)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4%}")
model.save("./mymodel.h5")

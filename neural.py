import tensorflow as tf
from tensorflow.keras import datasets, layers, models


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

class new_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')> 0.95):
            self.model.stop_training = True
            stop_epochs.append(epoch+1)

'''
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])
'''

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
results = []
stop_epochs =[]
for el in range (0, 10):


    callback = new_callback()
    '''
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])

    '''
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    #model.summary()
    model.compile(optimizer='adagrad',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=15,  callbacks=[callback],
                        validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    results.append(test_acc)

print(sum(stop_epochs)/10)
print(stop_epochs)
#print(sum(results)/10)
#print(results)





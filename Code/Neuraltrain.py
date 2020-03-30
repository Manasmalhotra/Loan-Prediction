import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
#kernel_regularizer=tf.keras.regularizers.l2(0.1)]

npz=np.load('Loan_data_train.npz')
train_inputs=npz['inputs'].astype(np.float)
train_targets=npz['targets'].astype(np.int)

npz=np.load('Loan_data_test.npz')
test_inputs=npz['inputs'].astype(np.float)
test_targets=npz['targets'].astype(np.int)

npz=np.load('Loan_data_validation.npz')
validation_inputs=npz['inputs'].astype(np.float)
validation_targets=npz['targets'].astype(np.int)

input_size=11
out_size=2
hidden_layer=50
early_stop=tf.keras.callbacks.EarlyStopping(patience=5)
model=tf.keras.Sequential([
                           tf.keras.layers.Dense(hidden_layer,activation='relu'),
                           tf.keras.layers.Dense(hidden_layer,activation='relu'),
                           tf.keras.layers.Dense(hidden_layer,activation='relu'),
                            tf.keras.layers.Dense(hidden_layer,activation='relu'),
                           tf.keras.layers.Dense(out_size,activation='softmax')
                           ])

opt=tf.keras.optimizers.SGD(lr=0.01)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
batch_size=10
epochs=100
model.fit(train_inputs,train_targets,batch_size=batch_size,epochs=100,validation_data=(validation_inputs,validation_targets),verbose=2)
test_loss,test_accuracy=model.evaluate(test_inputs,test_targets)
print(test_loss," ",test_accuracy)

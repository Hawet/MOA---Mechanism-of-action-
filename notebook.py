import pandas as pd
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
COLS = ['cp_type','cp_dose']
FE = []
for col in COLS:
    for mod in train_features[col].unique():
        FE.append(mod)
        train_features[mod] = (train_features[col] == mod).astype(int)
del train_features['sig_id']
del train_features['cp_type']
del train_features['cp_dose']
FE+=list(train_features.columns)
del train_targets['sig_id']
import tensorflow as tf
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(len(list(train_features.columns))))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(16000, activation="relu"))
model.add (tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(8000, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(4000, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2000, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(206, activation="softmax"))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9),loss='categorical_crossentropy',metrics='AUC')

#creatimg validation dataset

import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(np.array(train_features.to_numpy(), dtype=np.float),np.array(train_targets.to_numpy(), dtype=np.float),test_size=0.10, random_state = 42)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(100).batch(64)
model.fit(train_dataset,epochs=20,batch_size=10,validation_data=(X_val,Y_val))
#model.save('model.h5')

test_features = pd.read_csv("../input/lish-moa/test_features.csv")
#print(test_features)
#model = tf.keras.models.load_model("../input/lish-moa/model.h5")

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

test_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_features.to_numpy(), dtype=np.float))
test_dataset = test_dataset.batch(64)
COLS = ['cp_type','cp_dose']
for col in COLS:
    for mod in test_features[col].unique():
        test_features[mod] = (test_features[col] == mod).astype(int)

sig_id = pd.DataFrame()
sig_id = test_features.pop('sig_id')
del test_features['cp_type']
del test_features['cp_dose']

columns = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
del columns['sig_id']
predict = model.predict(test_features)
#print(predict.shape)
ids = pd.read_csv('../input/lish-moa/sample_submission.csv')
ids = ids['sig_id']
# creating submission for kaggle
submission = pd.DataFrame(data=predict,columns=columns.columns)
#print(submission)
submission.insert(0,column='sig_id',value=ids)
#print(submission)
submission.to_csv("submission.csv",index = False)



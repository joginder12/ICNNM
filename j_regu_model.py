
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
#import shap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, Dense,MaxPooling1D,Flatten, BatchNormalization
from sklearn.metrics import confusion_matrix 
from keras import backend as K
import os
#shap.initjs()

# List of input CSV file names
input_files = ['I_breast_subtype30.csv', 'I_grouped_n_tissue30.csv', 'I_Kidney_subtype20.csv','I_Lung_Subtype20.csv', 'I_Nor-Pan-cancer20.csv', 'I_Ntype_pan_can20.csv', 'I_Pan-can20.csv']

# Define a function to process each input file

# Directory where you want to save the output files# 
# Title: A Stacking Ensemble deep learning approach for cancer type classification based on TCGA data 
Stacked_folder = 'jreg_output'

# Create the output folder if it doesn't exist
if not os.path.exists(Stacked_folder):
    os.makedirs(Stacked_folder)
    
def process_input_file(input_file):
    # Read the input CSV file
    data = pd.read_csv(input_file, delimiter='\t')
    label_encoder = LabelEncoder().fit(data.miRNA_ID)
    labels = label_encoder.transform(data.miRNA_ID)
    classes = list(label_encoder.classes_)
    input = data.drop('miRNA_ID', axis=1)


    scaled_input = input.values


    nb_features = scaled_input.shape[1] # 1 is for column dimension
    nb_class = len(classes)
    X_train, X_test, y_train, y_test = train_test_split(scaled_input, labels, test_size=0.33, random_state=50)
    X_train_r = np.reshape(X_train, (len(X_train), nb_features, 1))
    X_test_r = np.reshape(X_test, (len(X_test), nb_features, 1))
    y_train_r = to_categorical(y_train, nb_class)
    y_test_r = to_categorical(y_test, nb_class)

    background= X_train_r[np.random.choice(X_train_r.shape[0], replace=False)]
#print(background)
    nn_in = X_train.shape[1]
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))                         

    def mcc(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        
        numerator = (true_positives * true_negatives - false_positives * false_negatives)
        denominator = K.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))
    
        return numerator / (denominator + K.epsilon())

    def network(nn_in):####new code to compute
        model = Sequential()
        model.add(Conv1D(filters = 17, kernel_size=5, input_shape=(nb_features, 1)))
        model.add(BatchNormalization(batch_size=128))
        model.add(LeakyReLU())
        model.add(MaxPooling1D(pool_size=5))
        model.add(Dropout(0.15))

        model.add(Conv1D(filters = 48, kernel_size=2, input_shape=(nb_features, 1)))
        model.add(BatchNormalization(batch_size=128))
        model.add(LeakyReLU())
        model.add(MaxPooling1D(pool_size=5))
        model.add(Dropout(0.55))

        model.add(Conv1D(filters = 26, kernel_size=2, input_shape=(nb_features, 1)))
        model.add(BatchNormalization(batch_size=128))
        model.add(LeakyReLU())
        model.add(MaxPooling1D(pool_size=3))
        model.add(Dropout(0.40))

        model.add(Flatten())
        model.add(Dense(units=44,activation='LeakyReLU'))
        model.add(Dense(units=42,activation='LeakyReLU'))
        model.add(Dense(units=50,activation='LeakyReLU'))
        #model.add(Dropout(0.50))
        model.add(Dense(units=nb_class,kernel_regularizer=l2(0.01),activation='softmax'))
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy',f1_m,precision_m, recall_m, mcc])
        return model

    model=network(nn_in)
    
    nb_epoch = 50
    cvscores = []
    cvscores1 = []
    confusion_matrices = []
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
    i=1
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    for input_index, valid_index in sss.split(scaled_input, labels):
        print("\nRound: %d\n"% i)
        i=i+1
        X_input, X_valid = scaled_input[input_index], scaled_input[valid_index]
        y_input, y_valid = labels[input_index], labels[valid_index]
# reshape train data
        X_input_r = np.reshape(X_input, (len(X_input), nb_features, 1))
    # reshape validation data
        X_valid_r = np.reshape(X_valid, (len(X_valid), nb_features, 1))
        y_input_r = to_categorical(y_input, nb_class)
        y_valid_r = to_categorical(y_valid, nb_class)
        history=model.fit(X_input_r, y_input_r, epochs=nb_epoch, validation_data=(X_valid_r, y_valid_r),callbacks=[callback])
        scores = model.evaluate(X_valid_r, y_valid_r, verbose=0)
        print(len(scores))
        # print("\n%s: %.2f%% \n" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        cvscores1.append(scores[0] * 100)
        
        
        y_predict=model.predict(X_valid_r, batch_size=10, verbose=0)
        y_valid_r = np.argmax(y_valid_r, axis=1)
        y_predict= np.argmax(y_predict, axis=1)
        #print(y_valid_r, y_predict)
        confusion_matrices.append(confusion_matrix(y_valid_r, y_predict))
#print('Confusion Matrix\n')
#print(confusion)
#print(cvscores1)
# print("Training accuracy: ", history.history['accuracy'])
        hist_df = pd.DataFrame(history.history)
        output_file = f"{os.path.splitext(input_file)[0]}_stacked_{i-1}.csv"
        output_path = os.path.join(Stacked_folder, output_file)
        hist_df.to_csv(output_path, index=False)

        print(f'{input_file} processed and saved as {output_file}')

    del X_input, X_valid, y_input, y_valid, X_input_r, X_valid_r, y_input_r, y_valid_r, scores

for input_file in input_files:
    process_input_file(input_file)

print('All files processed and saved.')

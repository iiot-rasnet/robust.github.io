import  pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import  seaborn as sns
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import TSNE
import time
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Input, ZeroPadding1D
from tensorflow.keras.layers import MaxPooling1D, Add, AveragePooling1D
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix, classification_report, precision_score, recall_score
from sklearn.metrics import f1_score as f1_score_rep
import seaborn as sn
import keras.backend as K
import tensorflow as tf
from keras.metrics import Recall, Precision
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df=pd.read_csv('preprocessed_DNN.csv',low_memory=False)






feat_cols=list(df.columns)
label_col="Attack_type"

feat_cols.remove(label_col)




empty_cols = [col for col in df.columns if df[col].isnull().all()]

# corr_matrix=df[feat_cols].corr()


skip_list = ["icmp.unused", "http.tls_port", "dns.qry.type", "mqtt.msg_decoded_as"]
df.drop(skip_list,axis=1,inplace=True)
feat_cols=list(df.columns)
feat_cols.remove(label_col)

X=df.drop([label_col],axis=1)
y=df[label_col]
del df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

del X
del y


label_encoder =LabelEncoder()

y_train =  label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)

class_weights = {k: v for k,v in enumerate(class_weights)}
# print(class_weights)


min_max_scaler = MinMaxScaler()
X_train =  min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
X_tr=X_train
X_te=X_test
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

input_shape = X_train.shape[1:]

num_classes = len(np.unique(y_train))

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

def ROC_plot(y_true_ohe, y_hat_ohe, label_encoder, n_classes):
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_ohe[:, i], y_hat_ohe[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_ohe.ravel(), y_hat_ohe.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(20, 20))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(label_encoder.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("multiclass characteristic")
    plt.legend(loc="lower right")
    plt.show()


def print_score(y_pred, y_real, label_encoder):
    print("Accuracy: ", accuracy_score(y_real, y_pred))
    print("Precision:: ", precision_score(y_real, y_pred, average="micro"))
    print("Recall:: ", recall_score(y_real, y_pred, average="micro"))
    print("F1_Score:: ", f1_score_rep(y_real, y_pred, average="micro"))

    print()
    print("Macro precision_recall_fscore_support (macro) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="macro"))

    print()
    print("Macro precision_recall_fscore_support (micro) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="micro"))

    print()
    print("Macro precision_recall_fscore_support (weighted) average")
    print(precision_recall_fscore_support(y_real, y_pred, average="weighted"))

    print()
    print("Confusion Matrix")
    cm = confusion_matrix(y_real, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index=[i for i in label_encoder.classes_],
                         columns=[i for i in label_encoder.classes_])
    plt.figure(figsize=(12, 10))
    sn.heatmap(df_cm, annot=True,cmap='Oranges')
    plt.show()

    print()
    print("Classification Report")
    print(classification_report(y_real, y_pred, target_names=label_encoder.classes_))


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv1D(filters=F1, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv1D(filters=F3, kernel_size=1, strides=1, padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    # define name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv1D(filters=filters[0], kernel_size=1, strides=s, name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Attention mechanism
    attention = Dense(filters[0], activation='sigmoid', name=conv_name_base + 'attn')(X)
    X = Multiply()([X, attention])

    # Second component of main path
    X = Conv1D(filters=filters[1], kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2b')(X)

    X = Activation('relu')(X)
    attention = Dense(filters[1], activation='sigmoid', name=conv_name_base + 'attn1')(X)
    X = Multiply()([X, attention])

    # Third component of main path
    X = Conv1D(filters=filters[2], kernel_size=1, strides=1, name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base + '2c')(X)


    # Short path
    X_shortcut = Conv1D(filters=filters[2], kernel_size=1, strides=s, name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(name=bn_name_base + '1')(X_shortcut)

    # Add the two paths together
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X




def ResNet50(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding1D((3, 3))(X_input)

    X = Conv1D(filters=64, kernel_size=7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=3, strides=2)(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling1D(pool_size=2, padding='same')(X)
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

from keras.layers import Multiply
from keras.layers import dot
from keras.layers import Dense, Lambda, Permute, multiply, concatenate, Reshape
from keras.initializers import glorot_uniform
def attention_3d_block(hidden_states):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec', kernel_initializer=glorot_uniform(seed=0))(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    return context_vector

def build_model(num_classes, input_shape=(92, 1)):
    base_model = ResNet50(input_shape=input_shape)
    headModel = base_model.output
    headModel = Flatten()(headModel)
    headModel = Dense(256, activation='relu', name='fc1', kernel_initializer=glorot_uniform(seed=0))(headModel)
    headModel = Dense(128, activation='relu', name='fc2', kernel_initializer=glorot_uniform(seed=0))(headModel)

    # 添加注意力层
    hidden_states = Reshape((2, 2048))(base_model.output)
    context_vector = attention_3d_block(hidden_states)
    headModel = concatenate([context_vector, headModel], name='concatenate')

    headModel = Dense(num_classes, activation='softmax', name='fc3', kernel_initializer=glorot_uniform(seed=0))(
        headModel)
    model = Model(inputs=base_model.input, outputs=headModel)
    opt = Adam(learning_rate=0.002)
    model.compile(optimizer=opt, loss=tf.keras.metrics.categorical_crossentropy,
                  metrics=['accuracy', Recall(), Precision(), f1_score])
    return model


model = build_model(num_classes, input_shape=input_shape)

model_weights_file_path ="model/attention-change528.h5"
checkpoint = ModelCheckpoint(filepath=model_weights_file_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min", save_weights_only=True)
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode="min", verbose=1, min_lr=0)
# plotlosses = PlotLossesKeras()

call_backs = [checkpoint, early_stopping, lr_reduce]
EPOCHS = 50
BATCH_SIZE =256
#model.load_weights("model/attention-change5263.h5")

# # # for i in range(10):
# # model.fit(X_train, y_train,
# #               validation_data=(X_test, y_test),
# #               # validation_split=0.1
# #               epochs=EPOCHS,
# #               batch_size=BATCH_SIZE,
# #               callbacks=call_backs,
# #               class_weight=class_weights,
# #               verbose=1)
#     # y_hat = model.predict(X_test)
#     # y_hat = np.argmax(y_hat, axis=1)
#     # y_true = np.argmax(y_test, axis=1)
#     # report = classification_report(y_true, y_hat, output_dict=True)
#     # f1_scores = [report[str(i)]['f1-score'] for i in range(6)]
#     # x = 0
#     # for scores in f1_scores:
#     #     f1_scores[x] = 1 / (scores + K.epsilon())
#     #     x += 1
#     # print(f1_scores)
#     # for a in range():
#     #     class_weights[a] = class_weights[a] + f1_scores[a]
#     # print(class_weights)
#
#
#
#
#
#
y_hat = model.predict(X_test)
y_hat = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)
y_true_ohe = to_categorical(y_true, num_classes=num_classes)
y_hat_ohe =  to_categorical(y_hat, num_classes=num_classes)
ROC_plot(y_true_ohe, y_hat_ohe, label_encoder, num_classes)
print_score(y_hat, y_true, label_encoder)
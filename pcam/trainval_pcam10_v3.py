from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print (os.getcwd())
import pandas as pd
import h5py
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_gcnn.keras_gcnn.applications.densenetnew_v9 import GDenseNet

import tempfile
import numpy as np
import pytest
import logging
import datetime
import matplotlib.pyplot as plt

from numpy.testing import assert_allclose
# from tensorflow.keras import backend as k
from keras import backend as K
from keras import losses
from keras import metrics
from keras import optimizers
from keras.models import save_model, load_model
from keras.utils import np_utils
from keras.utils.test_utils import keras_test
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model
import tensorflow as tf
import pickle,argparse
from sklearn.metrics import roc_curve, auc
from pcam_utils import plot_figures



def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--batch_size",type=int,default=32,help='batch size')
    parser.add_argument("-n",'--nb_classes',type=int,default=1,help="optional number of classes to classify images into, only to be specified if `include_top` is True, and if no `weights` argument is specified.")
    parser.add_argument("-e",'--epochs',type=int,default=50,help="epochs")
    parser.add_argument("-a",'--wh',type=int,default=96,help="img_rows, img_cols")
    parser.add_argument("-c",'--img_channels',type=int,default=3,help="img channels")
    parser.add_argument("-d",'--depth',type=int,default=9,help="number or layers in the DenseNet")
    parser.add_argument("-db",'--nb_dense_block',type=int,default=5,help=" number of dense blocks to add to end")
    parser.add_argument("-gr",'--growth_rate',type=int,default=2,help="number of filters to add per dense block")
    parser.add_argument("-f",'--nb_filter',type=int,default=24,help="initial number of filters. -1 indicates initial number of filters will default to 2 * growth_rate")
    parser.add_argument("-dr","--dropout_rate",type=float,default=0.0,help="dropout rate")
    parser.add_argument("-cg","--conv_group",type=str,default='D4',help="C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.")
    parser.add_argument("-s","--save_path",type=str, default='/root/workspace/output/p4m/try10/v3/', help="save path")
    parser.add_argument("-dp","--data_path",type=str, default='/root/workspace/output/p4m/dataset/pcam2/', help="data path")
    parser.add_argument("-net","--network",type=str, default='GDenseNet', help="network")
    
    
    return parser.parse_args()

def load_data(data_path):

    
    x_train = h5py.File(os.path.join(data_path,'camelyonpatch_level_2_split_train_x.h5'),'r').get('x').value
    y_train = h5py.File(os.path.join(data_path,'camelyonpatch_level_2_split_train_y.h5'), 'r').get('y').value
    #本来维度是(32768,1，1，1)，如果要降维可以用squeeze
    y_train = np.squeeze(y_train, axis=1) 
    y_train = np.squeeze(y_train, axis=1) 
    y_train = np.squeeze(y_train, axis=1)
    

    
    x_valid = h5py.File(os.path.join(data_path,'camelyonpatch_level_2_split_valid_x.h5'),'r').get('x').value
    y_valid = h5py.File(os.path.join(data_path,'camelyonpatch_level_2_split_valid_y.h5'), 'r').get('y').value
    y_valid = np.squeeze(y_valid, axis=1)
    y_valid = np.squeeze(y_valid, axis=1)
    y_valid = np.squeeze(y_valid, axis=1)
    
    
    x_test = h5py.File(os.path.join(data_path,'camelyonpatch_level_2_split_test_x.h5'),'r').get('x').value
    y_test = h5py.File(os.path.join(data_path,'camelyonpatch_level_2_split_test_y.h5'), 'r').get('y').value
    y_test = np.squeeze(y_test, axis=1)
    y_test = np.squeeze(y_test, axis=1)
    y_test = np.squeeze(y_test, axis=1)



    meta_train = pd.read_csv(os.path.join(data_path,'camelyonpatch_level_2_split_train_meta.csv'))
    meta_valid = pd.read_csv(os.path.join(data_path,'camelyonpatch_level_2_split_valid_meta.csv'))
    meta_test = pd.read_csv(os.path.join(data_path,'camelyonpatch_level_2_split_test_meta.csv'))
    if K.image_data_format() == 'channels_first':
        raise NotImplementedError()
    
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    
    # 如果要使用"categorical_crossentropy"作为目标函数，y变成one-hot
    #y_train = np_utils.to_categorical(y_train, nb_classes)
    #y_valid = np_utils.to_categorical(y_valid, nb_classes)
    #y_test = np_utils.to_categorical(y_test, nb_classes)
    
    return (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test)

 
def plot_history(save_path, histories, key=['loss','acc', 'precision','recall']):
    """ plot model training history
    """

    fig, axes = plt.subplots(1, 4, figsize=(20, 5),dpi=100)

    for i, ax_key in enumerate(key):
        ax = axes[i]
        ax_key = key[i]

        for name, history in histories:
            val = ax.plot(history.epoch, history.history['val_' + ax_key], '--', label=name.title() + ' Val')
            ax.plot(history.epoch, history.history[ax_key], color=val[0].get_color(), label=name.title() + ' Train')

        ax.set_xlabel('Epochs')
        ax.set_ylabel(ax_key.replace('_', ' ').title())

        if ax_key == 'acc':
            ax.legend()
        else:
            ax.legend(loc=1)

    plt.show()

    fig.savefig(save_path + 'training-history.png', bbox_inches='tight')


    
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc





if __name__=='__main__':
    
    # 训练参数
    args=get_arg()
    batch_size = args.batch_size
    nb_classes = args.nb_classes
    epochs = args.epochs
    img_rows, img_cols = args.wh, args.wh
    img_channels = args.img_channels
    data_path = args.data_path
    
    
    # Parameters for the DenseNet model builder
    img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
        img_rows, img_cols, img_channels)
    depth = args.depth
    nb_dense_block = args.nb_dense_block
    growth_rate = args.growth_rate  # number of z2 maps equals growth_rate * group_size, so keep this small.
    nb_filter = args.nb_filter
    dropout_rate = args.dropout_rate  # 0.0 for data augmentation
    conv_group = args.conv_group # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
    network = args.network
    
    if not (conv_group=='D4' or conv_group=='C4'):
        print ('Please make sure -cg is from "C4" or "D4"')
    weights_file = network + '-' + conv_group + '-' + str(depth) +'-' +  str(nb_dense_block)+ '-' + 'PCAM.h5'
    print ('wights file name:',weights_file )
        
    # 其他Keras可修改的参数
    use_gcnn = True
    optimizer = Adam(lr=1e-3)  # 其他优化器 https://keras-cn.readthedocs.io/en/latest/other/optimizers/
    loss = 'binary_crossentropy'  #其他损失函数 https://keras-cn.readthedocs.io/en/latest/other/objectives/  
    metrics=['accuracy', precision, recall,auc]   # https://keras-cn.readthedocs.io/en/latest/legacy/other/metrics/#_6
    

    
    # 保存地址和结果
    save_path = args.save_path
    logtime = str(datetime.datetime.now())
    if not os.path.exists(save_path):
        os.system(f'mkdir -p {save_path}')
    logging.basicConfig(level=logging.INFO,filename=os.path.join(save_path, 'train_v3_log_' + logtime + '.txt'))
    logging.info(f'=====================Parameters==========================')
    logging.info(str(args))
    
    
    
    # Create the model (without loading weights)
    # include_top=False时默认的输入是224，224，3，在网络顶层加全连接层，并且不能设置pooling；
    model = GDenseNet(mc_dropout=False, include_top=True, padding='valid', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                  nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, 
                  pooling='avg',classes=nb_classes,depth=depth,
                  use_gcnn=use_gcnn, conv_group=conv_group)
    print('Model created')
    model.summary()
    
    with open(save_path + 'model.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    # 保存模型结构，用netron可视化
    model.save(save_path + 'pcam_model.h5') 
    #可视化模型
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file=save_path +'pcam_model.png',show_shapes=True)

    
    
    # model compile
    #model = multi_gpu_model(model, gpus=2)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print('Finished compiling')
    
    
    # load data
    (x_train, y_train, meta_train), (x_valid, y_valid, meta_valid), (x_test, y_test, meta_test) = load_data(data_path)
    print ('load data success')
    print ('x_train, y_train,x_valid, y_valid,x_test, y_test shape:',
           x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_test.shape)
    logging.info(f'=====================Data==========================')
    logging.info(f'x_train, y_train,x_valid, y_valid,x_test, y_test shape:')
    
    logging.info(f'{str(x_train.shape)},{str(y_train.shape)},{str(x_valid.shape)},{str(y_valid.shape)},{str(x_test.shape)},{str(y_test.shape)}')
    # 构建batch
    generator = ImageDataGenerator(
              preprocessing_function=lambda x: x/255.)
              #width_shift_range=4,  # randomly shift images horizontally
              #height_shift_range=4,  # randomly shift images vertically 
              #horizontal_flip=True,  # randomly flip images
              #vertical_flip=True)  # randomly flip images

    
    
    # 拿几个数据测试下是否可以正常跑
    print (x_train[0:5].shape, y_train[0:5].shape)
    model.train_on_batch(x_train[0:5], y_train[0:5])
    print ('model can run')
    
    
    # callback: https://keras.io/zh/callbacks/
    ## tensorboard 可视化
    call_tensorboard = TensorBoard(log_dir=os.path.join(save_path,'logs'),  # log 目录
                 histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=batch_size,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=True, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
    ## 当标准评估停止提升时，降低学习速率。
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=20, min_lr=0.5e-6)
    ## 当被监测的数量不再提升，则停止训练。
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=100)
    ##在每个训练期之后保存模型
    weightpath=save_path + "weights.best.hdf5"
    model_checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', save_best_only=True,
                                   save_weights_only=True, mode='auto')
    
    ## train
    callbacks = [lr_reducer, early_stopper, model_checkpoint]
    print ('training')
    history = model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) // batch_size,
                    epochs=epochs,
                    validation_data=generator.flow(x_valid, y_valid, batch_size=batch_size), validation_steps=len(x_valid) // batch_size,
                    callbacks=callbacks,
                    verbose=1)
    logging.info(f'=====================Train==========================')
    print(history.history)
    print(history.epoch)
    logging.info(f'{str(history.history)},{str(history.epoch)}')
    #save history
    last_score = {}
    for key in history.history.keys():
        last_score[key] = history.history[key][-1]
    logging.info(f'{str(last_score)}')
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    history_json_file = save_path +  'history.json' 
    with open(history_json_file, mode='w') as f:
        hist_df.to_json(f)

    plot_history(save_path,[(weights_file.rstrip('.h5'), history)])
    
    
    # evaluate
    try:
        model.load_weights(save_path + "weights.best.hdf5")
    except:
        print ('no best model,please train again')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    score = model.evaluate_generator(generator.flow(x_test, y_test, batch_size=1),steps=32768)
    print('Test loss: '+str(score[0]))
    print('Test accuracy: '+str(score[1]))
    print('Test auc: '+str(score[-1]))
    logging.info(f'Test loss/Test accuracy/precision/recall/auc: ')
    logging.info(str(score))

    #y_pred = model.predict(x_test)
    y_pred = model.predict_generator(generator.flow(x_test, batch_size=1),steps=32768)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print("ROC auc: "+str(roc_auc))
    logging.info(f'roc_auc')
    logging.info(str(roc_auc))

    logging.info("plotting figures")

    plot_figures(fpr, tpr, history, roc_auc, save_path + "roc.png", save_path + "loss.png", save_path+ "accuracy.png")
    





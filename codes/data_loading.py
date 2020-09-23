import pandas as pd
import numpy as np
from random import shuffle
import cv2

def get_csv_data(path):

    '''

    helps to return dataframes of specific categories

    input path of the annotated csv file
    output a dataframe of fandr,f,r,perfect
    '''

    data = pd.read_csv(path)

    #both repetitions and fillers
    fandr = data[data['filler'] == 1]
    fandr = fandr[fandr['repetition'] == 1]

    #only fillers
    f = data[data['repetition'] == 0]
    f = f[f['filler'] == 1]

    #only repetitions
    r = data[data['filler'] == 0]
    r = r[r['repetition'] == 1]

    #only perfect sentences
    perfect = data[data['filler'] == 0]
    perfect = perfect[perfect['repetition'] == 0]

    return (fandr,f,r,perfect)

def splitting(df):
  '''
  input enter the data frame
  helps to split dataframe to train and test with a ratio of 80:20
  '''
  msk = np.random.rand(len(df)) < 0.8
  train = df[msk]
  test = df[~msk]
  return train,test


def train_test_val_split(df):
    '''
    input enter the dataframe
    output return the splitting of dataframe to train,test,validation dataframe sets as a tuple
    '''
    train_df    ,test_df     = splitting(df)
    train_df    ,val_df     = splitting(train_df)
    return (train_df,test_df,val_df)

def get_data(perfect,f,r,fandr,batch=8,size = (256*8,256),root_path = '/content/spectrograms/'):
    while True:
        imp_data = []
        count = 0

        def read_image(file_name):
            img = cv2.imread(file_name,0)
            img =cv2.resize(img,size)
            img = np.expand_dims(img, axis=-1)
            return img

        perfect = perfect.sample(frac=1)
        f = f.sample(frac=1)
        r = r.sample(frac =1)
        fandr = fandr.sample(frac = 1)

        # print(fandr.head(5))

        for i in np.array(fandr.sample(frac=1).head(batch//4)):
            image = read_image(root_path+i[0])
            labels = list(i[1:])
            imp_data.append([image,labels])
            # print('fandr')
        
        for i in np.array(r.sample(frac=1).head(batch//4)):
            image = read_image(root_path+i[0])
            labels = list(i[1:])
            imp_data.append([image,labels])
            # print('f')


        for i in np.array(f.sample(frac=1).head(batch//4)):
            image = read_image(root_path+i[0])
            labels = list(i[1:])
            imp_data.append([image,labels])
            # print('r')


        for i in np.array(perfect.sample(frac=1).head(batch//4)):
            image = read_image(root_path+i[0])
            labels = list(i[1:])
            imp_data.append([image,labels])
            # print('perfect')

        shuffle(imp_data)
        # imp_data = np.array(imp_data)

        # print(imp_data.shape)
        
        images = [i[0] for i in imp_data]
        # print(np.array(images).shape)
        labels = [i[1] for i in imp_data]

        yield (np.array(images)/255,np.array(labels))




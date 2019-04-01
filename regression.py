import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
import cv2
import tensorflow as tf
import numpy as np
trainingData = list();
trainingLabels = list()
trainingLabels1 = list()

validationLabels = list()
learning_rate = 0.001


def adjust_gamma(image, gamma):
    invGamma = 1.0/gamma
    table = np.array([((i/255)** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image, table)

def loadData():
    with open('/data/schugh/image-processing/set_office_crop_64/data.txt','r') as f:
        trainData = list()
        trainLabels = list()
        trainLabels1 = list()
        validData = list()
        validLabels = list()
        validLabels1 = list()
        reader = csv.reader(f)
        for row in reader:
            if row[1] == 'train':
                if row[2] == 'left' or row[2] == 'right':
                    trainData.append(row[0])
                    trainLabels.append(row[3])
                    trainLabels1.append(row[4])
            elif row[1] == 'validation':
                if row[2] == 'left' or row[2] == 'right':
                    validData.append(row[0])
                    validLabels.append(row[3])
                    validLabels1.append(row[4])

    return trainData, trainLabels, trainLabels1, validData, validLabels,validLabels1

def convertOneHotTrain(trainTarget, trainTarget1):
    newtrain = np.zeros((trainTarget.shape[0], 2))
    for item in range(0, trainTarget.shape[0]):
        newtrain[item][0] = trainTarget[item]
        newtrain[item][1] = trainTarget1[item]
    return newtrain

def convertOneHotValid(validTarget, validTarget1):
    newvalid = np.zeros((validTarget.shape[0], 2))
    for item in range(0, validTarget.shape[0]):
        newvalid[item][0] = validTarget[item]
        newvalid[item][1] = validTarget1[item]
    return newvalid

def conv2d(x, W, b, name,strides=1):
    x = tf.nn.conv2d(input = x, filter = W, strides=[1, strides, strides, 1], padding='SAME', name = name)
    x = tf.nn.bias_add(x, b)
    x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)

def maxpool2d(x,name,k=2):
    return tf.nn.avg_pool(value = x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME', name = name)

def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 64, 64,1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],"Convolution1")
    #conv1 = tf.nn.dropout(conv1,0.2)
    conv1 = maxpool2d(conv1,"Pooling",2)
    print (conv1.shape)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'],"Convolution2")
    print (conv2.shape)
    #conv2 = tf.nn.dropout(conv2,0.2)
    conv2 = maxpool2d(conv2,"Pooling",2)
    print (conv2.shape)
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'],"Convolution3")
    #conv3 = tf.nn.dropout(conv3,0.2)
    conv3 = maxpool2d(conv3,"Pooling",2)
    print (conv3.shape)
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'],"Convolution4")
    #conv4 = tf.nn.dropout(conv4,0.2)
    conv4 = maxpool2d(conv4,"Pooling",2)
    print (conv4.shape)
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'],"Convolution4")
    #conv5 = tf.nn.dropout(conv5,0.2)
    print (conv5.shape)
    conv5 = maxpool2d(conv5,"Pooling",2)
    print (conv5.shape)
    fc1 = tf.contrib.layers.flatten(conv5)
    #fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc2 = tf.matmul(fc1, weights['wd1']) +  biases['bd1']
    fc2 = tf.layers.batch_normalization(fc2)
    fc2 = tf.nn.relu(fc2,name ='finalrelu')
    print (fc2.shape)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, 0.5,name = 'Dropout')
    #fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    #print fc2.shape
    #fc2 = tf.matmul(fc2, weights['wd2']) +  biases['bd2']
    #fc2 = tf.nn.relu(fc2,name ='finalrelu1')
    #fc2 = tf.layers.batch_normalization(fc2)
    # Output, class prediction
    logits = tf.matmul(fc2, weights['out']) + biases['bout']
    print (logits.shape)
    return logits

def main():
    trainingDatafilename, trainingLabels, trainingLabels1, validationDatafilename, validationLabels,validationLabels1 = loadData()
    trainingLabels2 = convertOneHotTrain(np.array(trainingLabels),np.array(trainingLabels1))
    validationLabels2 = convertOneHotTrain(np.array(validationLabels),np.array(validationLabels1))
    validationData = list()
    sess = tf.Session()
    initializer = tf.contrib.layers.xavier_initializer()
    with sess.as_default():
        writer = tf.summary.FileWriter('./graphs/' + "regression",sess.graph)
        weights = {
            'wc1': tf.get_variable("wc1", shape=[3, 3, 1, 16],initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable("wc2", shape=[3, 3,16, 32],initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable("wc3", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer()),
            'wc4': tf.get_variable("wc4", shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer()),
            'wc5': tf.get_variable("wc5", shape=[3, 3, 128, 128],initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable("wd1", shape=[2*2*128, 2048],initializer=tf.contrib.layers.xavier_initializer()),
            #'wd2': tf.get_variable("wd2", shape=[2048, 2048],initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable("out", shape=[2048,2],initializer=tf.contrib.layers.xavier_initializer())
        }
        biases = {
            'bc1': tf.get_variable("bc1", shape=[16],initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable("bc2", shape=[32],initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable("bc3", shape=[64],initializer=tf.contrib.layers.xavier_initializer()),
            'bc4': tf.get_variable("bc4", shape=[128],initializer=tf.contrib.layers.xavier_initializer()),
            'bc5': tf.get_variable("bc5", shape=[128],initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable("bd1", shape=[2048],initializer=tf.contrib.layers.xavier_initializer()),
            #'bd2': tf.get_variable("bd2", shape=[2048],initializer=tf.contrib.layers.xavier_initializer()),
            'bout': tf.get_variable("bout", shape=[2],initializer=tf.contrib.layers.xavier_initializer())
        }
        X = tf.placeholder(tf.float32, [None, 64, 64,1], name='input')
        Y = tf.placeholder(tf.float32, [None, 2])
        logits = conv_net(X, weights, biases)
        #loss = tf.reduce_mean(tf.squared_difference(Y, logits))
        delta_com = tf.subtract(Y, logits)
        norm_com = tf.norm(delta_com, axis=1)
        loss = tf.reduce_mean(norm_com)
        #loss = tf.norm(Y-logits,ord='euclidean')
        #loss = tf.losses.mean_squared_error(labels = Y , predictions = logits)
        #reg1 = tf.multiply(0.1, tf.reduce_sum(tf.square(weights['out'])))
        #reg2 = tf.multiply(0.1, tf.reduce_sum(tf.square(weights['wd2'])))
        #reg3 = tf.multiply(0.1 / 2, tf.reduce_sum(tf.square(weights['wd1'])))
        #reg4 = tf.multiply(0.1, tf.reduce_sum(tf.square(weights['wc5'])))
        #reg5 = tf.multiply(0.1, tf.reduce_sum(tf.square(weights['wc4'])))
        #reg6 = tf.multiply(0.1, tf.reduce_sum(tf.square(weights['wc3'])))
        #reg7 = tf.multiply(0.1, tf.reduce_sum(tf.square(weights['wc2'])))
        #reg8 = tf.multiply(0.1, tf.reduce_sum(tf.square(weights['wc1'])))
        totalLoss = loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(totalLoss)
        init = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        trainingLossList = list()
        validationLossList = list()
        normalizedImg = np.zeros((64, 64))
        print (np.array(validationDatafilename).shape)
        valid_dataset = tf.data.Dataset.from_tensor_slices(validationDatafilename)
        validLabels_dataset = tf.data.Dataset.from_tensor_slices(validationLabels2)
        train_dataset = tf.data.Dataset.from_tensor_slices(trainingDatafilename)
        test_dataset = tf.data.Dataset.from_tensor_slices(trainingLabels2)
        print (trainingLabels2.shape[0])
        print (validationLabels2.shape[0])
        for epoch in range(150):
            finaltrainingData = list()
            finalvalidationData = list()
            combindedTrainDataset = tf.data.Dataset.zip((train_dataset, test_dataset)).shuffle(trainingLabels2.shape[0]).batch(256)
            iterator = combindedTrainDataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(iterator.initializer)
            numberOfBatches = int(trainingLabels2.shape[0]/256)
            for i in range(numberOfBatches):
                val = sess.run(next_element)
                finaltrainingData = list()
                for image in (val[0]):
                    img = cv2.imread('/data/schugh/image-processing/set_office_crop_64/' + image.decode("utf-8"), 0)
                    #equ = cv2.equalizeHist(img)
                    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
                    cl1 = clahe.apply(img)
                    #equ = adjust_gamma(cl1,0.5)
                    normalizedImg1 = cv2.normalize(cl1,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                    normalizedImg1 = np.expand_dims(normalizedImg1, axis=2)
                    finaltrainingData.append(normalizedImg1)
                sess.run(optimizer, feed_dict={X:np.array(finaltrainingData),Y:val[1]})
            combindedValidDataset = tf.data.Dataset.zip((valid_dataset, validLabels_dataset)).shuffle(validationLabels2.shape[0]).batch(500)
            iterator1 = combindedValidDataset.make_initializable_iterator()
            next_element1 = iterator1.get_next()
            sess.run(iterator1.initializer)
            val1 = sess.run(next_element1)
            for i in val1[0]:
                img = cv2.imread('/data/schugh/image-processing/set_office_crop_64/' + i.decode("utf-8"), 0)
                clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
                cl1 = clahe.apply(img)
                #equ = adjust_gamma(cl1,0.5)
                #equ = cv2.equalizeHist(img)
                normalizedImg1 = cv2.normalize(cl1,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
                normalizedImg1 = np.expand_dims(normalizedImg1, axis=2)
                finalvalidationData.append(normalizedImg1)
            validationData1 = np.array(finalvalidationData)
            validationError = sess.run(totalLoss,feed_dict={X:validationData1,Y:val1[1]})*64
            print("validation error %g"%(validationError))
            finalvalidationData = list()
            trainingLossList.append(sess.run(totalLoss,feed_dict={X:np.array(finaltrainingData),Y:val[1]})*64)
            validationLossList.append(validationError)
        distributionError = list()
        for i in range(len(validationDatafilename)):
            initialData = list()
            img = cv2.imread('/data/schugh/image-processing/set_office_crop_64/' + validationDatafilename[i], 0)
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
            cl1 = clahe.apply(img)
            #equ = cv2.equalizeHist(img)
            #equ = adjust_gamma(cl1,0.5)
            normalizedImg1 = cv2.normalize(cl1,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
            normalizedImg1 = np.expand_dims(normalizedImg1, axis=2)
            finalvalidationData.append(normalizedImg1)
            initialData.append(normalizedImg1)
            distributionError.append(sess.run(totalLoss, feed_dict={X:initialData, Y:np.expand_dims(validationLabels2[i], axis = 0)})*64)
        validationData1 = np.array(finalvalidationData)
        validationError = sess.run(totalLoss,feed_dict={X:validationData1,Y:validationLabels2})*64
        print("Final validation error %g"%(validationError))
    distributionFinal = np.sort(distributionError)
    x = list()
    meanError = np.mean(distributionError)
    points = distributionFinal <= 1
    print ("Length less than 1 " + (str(len(distributionFinal[points]))))
    x.append((len(distributionFinal[points])))
    pointsList = [i for i, x in enumerate(points) if x]
    distributionFinal1 = np.delete(distributionFinal,pointsList)
    points = distributionFinal1 < np.mean(distributionError)
    print ("Length less than " + str(meanError) + " is " + str(len(distributionFinal1[points])))
    x.append((len(distributionFinal1[points])))
    pointsList = list()
    pointsList = [i for i, x in enumerate(points) if x]
    distributionFinal2 = np.delete(distributionFinal1, pointsList)
    points = distributionFinal2 <= np.mean(distributionError) + 1 
    print ("Length less than mean + 1 is " + str(len(distributionFinal2[points])))
    x.append(len(distributionFinal2[points]))
    pointsList = list()
    pointsList = [i for i, x in enumerate(points) if x]
    distributionFinal3 = np.delete(distributionFinal2, pointsList)
    #print (len(distributionFinal3))
    points = distributionFinal3 > np.mean(distributionError) + 1
    x.append(len(distributionFinal3[points]))
    #print (len(distributionFinal3[points]))
    print ("Length greater than mean + 1 is " + str(len(distributionFinal3[points])))
    plt.figure()
    label = ["Error<1","1<Error<Mean", "Mean<Error<Mean+1", "Error>Mean+1"]
    matplotlib.rcParams.update({'font.size': 6})
    plt.bar(label,x)
    plt.xlabel("Mean Pixel Error")
    plt.ylabel("Number of validation images")
    plt.savefig('histogram' + '.png')
    maxPoint = np.argmax(distributionError).astype(np.int64)
    minPoint = np.argmin(distributionError).astype(np.int64)
    print ("Min Error is " +  str(distributionError[minPoint]))
    print ("Min Filename is " + str(validationDatafilename[minPoint]))
    print ("Max Error is " + str(distributionError[maxPoint]))
    print ("Max Filename is "+str(validationDatafilename[maxPoint]))
    plt.figure()
    matplotlib.rcParams.update({'font.size':12})
    plt.plot(trainingLossList, 'r')
    plt.plot(validationLossList, 'b')
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean Pixel Error")
    plt.gca().legend(('training Loss','validation Loss'))

    plt.savefig('cnn_error_' + '.png')
    plt.figure()
    plt.plot(distributionFinal)
    plt.xlabel("Validation Images")
    plt.ylabel("Mean Pixel Error")
    plt.savefig("distributionError" + ".png")
if __name__ == '__main__':
    main()












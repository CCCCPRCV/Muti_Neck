# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import pandas as pd

if __name__ == '__main__':
    ## step 1: 加载数据
    print("step 1: load data...")
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    file_folder = '../collected_center/teacher1/'
    write_file = '../collected_center/teacher1_centers.txt'
    write_file_hander = open(write_file,mode='a')
    for file in os.listdir(file_folder):
        filename = file_folder + file
        fileIn = open(filename)
        count_datanumber = 0
        dataSet = []
        for line in fileIn.readlines():
            lineArr = eval(line.strip())
            dataSet.append(lineArr)
            count_datanumber +=1
        fileIn.flush()
        fileIn.close()
        # 设定不同k值以运算
        data = np.array(dataSet)
        data = data
        data_pd = pd.DataFrame(data)
        clf = KMeans(n_clusters=1)  # 设定k  ！！！！！！！！！！这里就是调用KMeans算法
        s = clf.fit(data)  # 加载数据集合
        class_center = clf.cluster_centers_
        class_center = np.squeeze(class_center)
        class_name = file.split('.')[0]
        class_index = classes.index(class_name)
        write_info = str(class_center.tolist()) + '|' + str(class_index)
        write_file_hander.write(write_info + ' \n')
        del dataSet
    write_file_hander.close()

        # predict_data = np.array([[0.10843329131603241, 0.14901815354824066, 0.13218678534030914, 0.13417311012744904, 0.12169473618268967, 0.11708094924688339, 0.1149507462978363, 0.10587126761674881, 0.1825885772705078, 0.1702050119638443, 0.21522067487239838, 0.19480662047863007, 0.14941756427288055, 0.1318892538547516, 0.07606912404298782, 0.18685215711593628, 0.1884501725435257, 0.24656563997268677, 0.19111241400241852, 0.10763281583786011, 0.08274222910404205, 0.08847364783287048, 0.20268268883228302, 0.22705353796482086, 0.2375718057155609, 0.1379452645778656, 0.07694971561431885, 0.06524892896413803, 0.11237792670726776, 0.18210411071777344, 0.2037963718175888, 0.17345894873142242, 0.1076173260807991, 0.07102000713348389, 0.06103014945983887, 0.09795959293842316, 0.13670113682746887, 0.148057758808136, 0.12800495326519012, 0.11260134726762772, 0.08712022751569748, 0.06394802033901215, 0.11248068511486053, 0.1306794285774231, 0.13372398912906647, 0.12079354375600815, 0.10637004673480988, 0.08268766850233078, 0.06469667702913284]] )
        # result = clf.predict(predict_data)
        # print(result.inertia_)


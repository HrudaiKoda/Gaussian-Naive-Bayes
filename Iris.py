import csv
import numpy as np
from math import sqrt
from math import pi
from math import exp
import random
from numpy import array
from sklearn.model_selection import KFold


def str_to_int_class(data):
    lab = data[:,-1]
    vals = dict()
    count = 0
    for i in lab:
        if(i in vals):
            #Do nothing
            u = 1
        else:
            vals[i] = count
            count = count + 1
    for i in range(len(data)):
        data[i,-1] = vals[data[i,-1]]
    return data
def load_dataset():
    data_set = list()
    with open('Iris.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            data_set.append(row)
    split = 0.65
    data_set = data_set[1:]
    jk = len(data_set)*split
    jk = int(jk)
    data = data_set
    print("lenght is {0}".format(len(data)))
    data = np.array(data)
    data = str_to_int_class(data)
    data = data[:,1:]
    data = data.astype("float")

    test_data = data_set[3::6]
    test_data = np.array(test_data)
    test_data = str_to_int_class(test_data)
    test_data = test_data[:,1:]
    test_data = test_data.astype("float")
    return data , test_data

def know_classes(data):
    sep_class = dict()
    for i in range(len(data)):
        temp  = data[i]
        lab = temp[-1]
        if(lab in sep_class):
            sep_class[lab].append(temp)
        else:
            sep_class[lab] = list()
    return sep_class

def summary_dataset(data):
    size = len(data[0])
    tot = len(data)
    summ_data = list()
    for i in range(size-1):
        mean = sum(data[:,i])/tot
        sd = (data[:,i] - mean)**2
        sd = sum(sd)/tot-1
        summ_data.append([mean,sd])
    return summ_data

def summary_class(sep_class):
    classes = list()
    summ_class = list()
    for key in sep_class:
        data_1 = sep_class[key]
        data_1 = np.array(data_1)
        size = len(data_1[0])
        tot = len(data_1)
        classes.append(key)
        summ_data_1 = list()
        for i in range(size-1):
            mean = sum(data_1[:,i])/tot
        
            sd = np.std(data_1[:,i])
            summ_data_1.append([mean,sd])
        summ_class.append(summ_data_1)
    return summ_class,classes

def probal(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
    

def calculate(summ_class,row,classes):
    prob = dict()
    for i in range(len(summ_class)):
        io = len(sep_class[classes[i]])
        prob[classes[i]] = io/float(75)
        for j in range(len(summ_class[0])):
            m ,s = summ_class[i][j]
            prob[classes[i]] *= probal(row[j],m,s)
        
    return prob

def test_model(summ_class,test_data,classes):
    labels = test_data[:,-1]
    acc = 0
    s = len(test_data)
    for i in range(int(s)):
        p = calculate(summ_class,test_data[i],classes)
        
        ke = max(p, key=p.get)
        if(labels[i] == ke):
            acc = acc + 1
    res = (acc/s)*100
    print(str(res)+ "%")


data ,test_data = load_dataset()
k_fold = data

kfold = KFold(3, True, 1)
co = 0
for train, test in kfold.split(k_fold):
     
	#print('train: %s, test: %s' % (k_fold[train], k_fold[test]))
    sep_class = know_classes(k_fold[train])
    summ_data = summary_dataset(k_fold[train])
    summ_class,classes = summary_class(sep_class)
    print("Fold----{0}".format(co))
    print("Train Accuracy", end= " ")
    test_model(summ_class,k_fold[train],classes)
    print("Test Accuracy " ,end = " ")
    test_model(summ_class,k_fold[test],classes)
    co = co +1






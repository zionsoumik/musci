import csv
import numpy as np
with open('wr.txt', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)
#print(your_list)
my_list=[]
for p in your_list:
    for l in p:
        my_list.append(l[:50])
#print(my_list)
#print(len(my_list[0]))
labels=np.empty((0,50))
for s in my_list:
    #print(labels)
    k=list(s)
    #print(len(k))
    h=np.array(k)
    h=h.astype(np.int)
    #print("h=",h)
    labels = np.vstack([labels, h])

#print("labels=",labels)
root = "C:/Users/deyso/PycharmProjects/sound/mp3folder"
dest = root + '/npy_files_TOTAL_train/' + 'labels.npy'
np.save(dest, labels)
import os
import struct 

def readFile(dir, filePath):
    imagelist = open(os.path.join(dir,filePath), 'r')

    dataset = []
    label_list = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        data_example['filename'] = dir + info[0]

        label_path = os.path.join(dir,info[0])
        print("label_path:",label_path)
        label_name = os.path.splitext(label_path)
        print("label_name:",label_name)
        label_name = label_name[0] + '.dat1'
        print("label_name:",label_name)
        f = open(label_name, 'rb')
        labels = f.read()
        
        for i in range(2,258):
        	labels_, = struct.unpack('f', labels[i*4:i*4+4])
        	label_list.append(labels_)
        	
        print("label_list:",label_list)
        print("\n")
        data_example['label'] = label_list
        dataset.append(data_example)
    print ("dataset:",dataset)
    return dataset

if __name__ == '__main__':
	dir = "G:/07_FRGC"
	filePath = "list2.txt"
	readFile(dir, filePath)
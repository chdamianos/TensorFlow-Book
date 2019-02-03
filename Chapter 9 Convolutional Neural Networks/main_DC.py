import numpy as np
import CnnClass
import cifar_tools_DC

names, data, labels = \
    cifar_tools_DC.read_data(
        './cifar-10-batches-py')

cnnModel = CnnClass.Cnn(input_dim=24, output_dim=10, epoch=10)
cnnModel.train(data, labels, './savedModel/model.ckpt')
for i in np.random.randint(0, high=len(labels), size=10):
    res = cnnModel.inference(data[i, :], './savedModel/model.ckpt')
    print(np.argmax(res), labels[i])
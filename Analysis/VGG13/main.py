import numpy as np
from Classification.CNNModel import VGG13
from matplotlib import pyplot as plt

segmented_train_data = np.load("../input/data-preparation-for-multiclass-classification/segmented_train_data.npy")
segmented_val_data = np.load("../input/data-preparation-for-multiclass-classification/segmented_val_data.npy")
segmented_test_data = np.load("../input/data-preparation-for-multiclass-classification/segmented_test_data.npy")


train_data = np.load("../input/data-preparation-for-multiclass-classification/train_data.npy")
val_data = np.load("../input/data-preparation-for-multiclass-classification/val_data.npy")
test_data = np.load("../input/data-preparation-for-multiclass-classification/test_data.npy")

train_target = np.load("../input/data-preparation-for-multiclass-classification/train_target.npy")
val_target = np.load("../input/data-preparation-for-multiclass-classification/val_target.npy")
test_target = np.load("../input/data-preparation-for-multiclass-classification/test_target.npy")

print(f"Total image for training data: {train_data.shape[0]}")
print(f"Total image for validation: {val_data.shape[0]}")
print(f"Total image for testing: {test_data.shape[0]}")

model = VGG13(input_shape=(128,128,1))

history = model.fit(train_data,train_target,epochs=50,validation_data=(val_data,val_target))

print(model.evaluate(test_data,test_target,batch_size=1))

plt.plot(history.history['loss'],'r',label = 'training loss')
plt.plot(history.history['val_loss'],label = 'validation loss')
plt.grid()
plt.xlabel('# Epoches')
plt.ylabel("loss")
plt.legend()
plt.savefig('loss.pdf') 

plt.plot(history.history['accuracy'],'r',label = 'training accuracy')
plt.plot(history.history['val_accuracy'],label = 'validation accuracy')
plt.xlabel('# Epoches')
plt.ylabel("accuracy")
plt.grid()
plt.ylim(0.10,1)
plt.xlim(0,20)
plt.legend()
plt.savefig('accuracy.pdf') 
plt.show()


from sklearn import metrics
pred = model.predict(test_data)
matrix = metrics.confusion_matrix(test_target.argmax(axis=1), pred.argmax(axis=1))
print(matrix)

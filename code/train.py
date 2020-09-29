import os
import pickle
import matplotlib.pyplot as plt
from data import DataFrame, ImagesData
from model import RCNN
import tensorflow as tf

# General
data_path = "../data/in/data/"  # Where is your input data
work_path = "../data/out/"  # Where everything will be saved
seed = 1338  # Random seed # 1337
verbose = 1  # 0: no output; 1: normal informations; 2: e v e r y th i n g

# DataFrame
dataframe_pickle_path = os.path.join(work_path, "dataframe.pickle")  # Where will the DataFrame be saved
force_preparation = True  # Do you want to bypassed the saved DataFrame
subsamples = -1  # Number of samples to use for the DataFrame; -1: Use all of them

# ImagesData
imagesdata_pickle_path = os.path.join(work_path, 'imagesdata.pickle')  # Where will the ImagesData be saved
number_of_results = 2000  # How many samples will selective search use
iou_threshold = 0.85  # What is the percent of precision required
max_samples = 15  # How many class samples do you want per image in the Selective Search
show_infos = True  # Show information for images output
show_labels = True  # Show labels for images output

# RCNN
model_and_weights_path = "../data/out/"  # Where will the model and weights be saved/loaded
loss = None  # Loss function; None: Use crossentropy
opt = None  # Optimization function; None: Use Adame
lr = 0.0001  # Learning rate #0.001 0.0001
epochs = 20  # Number of epochs 20
batch_size = 128 # 64
split_size = 0.10  # Test/Train proportion
checkpoint_path = os.path.join(work_path, 'checkpoint.h5')  # Where will the checkpoints be saved; None: No checkpoint (don't.)
early_stopping = False  # Should the learning stop if no more improvment is done
threshold = 0.80  # Threshold used for the recognition

dataframe = DataFrame(data_path, pickle_path=dataframe_pickle_path)
dataframe.prepare_data(force_preparation=force_preparation, subsamples=subsamples, verbose=verbose)
dataframe.summary()

if os.path.isfile(imagesdata_pickle_path):  # This shall be added in a future version directly in the package.
    with open(imagesdata_pickle_path, 'rb') as fi:
        imagesdata = pickle.load(fi)
else:
    imagesdata = ImagesData(dataframe, pickle_path=imagesdata_pickle_path)
    # That part is quite long, beware!
    imagesdata.prepare_images_and_labels(number_of_results=number_of_results, iou_threshold=iou_threshold,max_samples=max_samples, verbose=verbose)
    
    # Save it for later.
    with open(imagesdata_pickle_path, 'wb') as fi:
        pickle.dump(imagesdata, fi)

print('NB CLASSES : ' + str(imagesdata.get_num_classes()))

tf.debugging.set_log_device_placement(True)

strategy = tf.distribute.MirroredStrategy() # run on multiple GPUs
with strategy.scope():


	arch = RCNN(imagesdata, loss=loss, opt=opt, lr=lr, verbose=verbose)
	arch.train(epochs=epochs, batch_size=batch_size, split_size=split_size, checkpoint_path=checkpoint_path,
				early_stopping=early_stopping, verbose=verbose)
	
	#arch.model.save(filepath = '../data/out_test/test2/model.h5', save_format = 'h5')
	#tf.io.write_graph(arch.model.tf.graph, '../data/out_test/', 'graph.pbtxt')

loss = arch.history()['loss']
val_loss = arch.history()['val_loss']
accuracy = arch.history()['accuracy']
val_accuracy = arch.history()['val_accuracy']
# Loss
plt.plot(loss)
plt.plot(val_loss)
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.show()
print("Final loss: {}".format(loss[-1]))
# Accuracy
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()
print("Final accuracy: {}".format(accuracy[-1]))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Loss', color=color)
ax1.plot(val_loss, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(val_accuracy, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


imagesdata.show_image(6, show_infos=show_infos, show_labels=show_labels)
arch.test_image(6, show_infos=show_infos, show_labels=show_labels, number_of_results=number_of_results,threshold=threshold, verbose=verbose)

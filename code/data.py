import os
import pandas as pd
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
import cv2
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


class DataFrame:
    '''
    DataFrame object contains condensed informations from the annotations.
    '''
    def __init__(self, data_path, pickle_path=None):
        '''
        Define a new DataFrame.
        :param data_path: Folder to the data. Requires both `.jpg` and `.xml`, cf. README.
        :param pickle_path: Where to save the pickled DataFrame.
        '''
        self.data_path = data_path
        if pickle_path is None:
            self.pickle_path = os.path.join(data_path, 'dataframe.pickle')
        else:
            self.pickle_path = pickle_path
        self.df = None
        self.files = None
        self.classes = None

    def prepare_data(self, force_preparation=False, subsamples=-1, verbose=1):
        '''
        Data preparator.
        :param force_preparation: If we want to enforce data preparation.
        :param subsamples: If `subsamples > -1`, will only use `subsamples` datapoints.
        :param verbose: 0:=no output, 1:=errors only, 2:= everything.
        '''
        if not os.path.exists(self.pickle_path) or force_preparation:  # If we need to convert data
            # Collect files
            names = []
            for root, dirs, files in os.walk(self.data_path):
                for name in files:
                    if name.split('.')[-1] == 'xml':
                        basename = '.'.join(name.split('.')[:-1])

                        # Check if we have the corresponding .jpg file
                        if os.path.exists(os.path.join(root, basename + '.jpg')):
                            out = os.path.join(root, basename)
                            names.append(out)
                            if verbose == 2:
                                print(out)
                        else:
                            if verbose:
                                print("Error with file {}: .jpg does not exists.".format(basename))

            # Columns
            columns = ['file_name', 'class_name', 'xmin', 'ymin', 'xmax', 'ymax']

            # Data
            data = []
            for name in names:
                xml_path = name + '.xml'
                try:
                    tree = ElementTree.parse(xml_path)
                except ElementTree.ParseError:  # The annotation file is missing.
                    if verbose:
                        print("Error with file {}: Error while parsing.".format(xml_path))
                    continue
                root = tree.getroot()

                for obj in root.findall('object'):
                    temp = [name + '.jpg', obj.find('name').text]
                    for child in obj.find('bndbox'):
                        temp.append(child.text)
                    data.append(temp)

                if subsamples != -1:
                    if subsamples > 1:
                        subsamples -= 1
                    else:
                        break

            # Create a new pandas dataframe
            self.df = pd.DataFrame(data, columns=columns)
            if verbose == 2:
                print("*"*13)
                print(self.df.head())

            # Save pickle
            self.df.to_pickle(self.pickle_path)
        else:  # The data is already available
            self.df = pd.read_pickle(self.pickle_path)

        self.files = self.df.groupby('file_name').size()\
            .reset_index(name='counts')
        self.classes = self.df.groupby('class_name').size()\
            .reset_index(name='counts').sort_values(by='counts', ascending=False)

    # Why use len(foo.index)? https://stackoverflow.com/a/15943975
    def get_num_files(self):
        '''
        :return: Number of files.
        '''
        return len(self.files.index)

    def get_num_classes(self):
        '''
        :return: Number of classes.
        '''
        return len(self.classes.index)

    def get_num_pod(self):
        '''
        :return: Number of points of data.
        '''
        return len(self.df.index)

    def summary(self):
        '''
        Print a nice little summary of the DataFrame.
        '''
        print("We have {} classes for {} files, total points of data is {}.".format(self.get_num_classes(),
                                                                                    self.get_num_files(),
                                                                                    self.get_num_pod()))


class ImagesData:
    '''
    ImagesData contains samples of training data from the images.
    '''
    def __init__(self, DataFrame, pickle_path=None):
        '''
        Defines a new Images Data.
        :param DataFrame: A non-empty DataFrame
        :param pickle_path: Where to save the pickled ImagesData.
        '''
        self.DataFrame = DataFrame
        if DataFrame.get_num_pod() == 0:
            print("Warning: DataFrame is empty.")
        self.df = self.DataFrame.df
        self.files = self.DataFrame.files
        if pickle_path is None:
            self.pickle_path = os.path.join(self.DataFrame.data_path, 'imagesdata.pickle')
        else:
            self.pickle_path = pickle_path
        self.images = None
        self.labels = None
        self.classes = []
        cv2.setUseOptimized(True)

    def get_iou(self, bbox_1, bbox_2):
        '''
        Intersection over Union (IoU).
        https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        Basically, Area of Intersection / Area of Union.
        :param bbox_1: Dict, {xmin, ymin, xmax, ymax}
        :param bbox_2: Dict, {xmin, ymin, xmax, ymax}
        :return:
        '''
        x_left = max(bbox_1['xmin'], bbox_2['xmin'])
        y_top = max(bbox_1['ymin'], bbox_2['ymin'])
        x_right = min(bbox_1['xmax'], bbox_2['xmax'])
        y_bottom = min(bbox_1['ymax'], bbox_2['ymax'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bbox_1['xmax'] - bbox_1['xmin']) * (bbox_1['ymax'] - bbox_1['ymin'])
        bb2_area = (bbox_2['xmax'] - bbox_2['xmin']) * (bbox_2['ymax'] - bbox_2['ymin'])

        return intersection_area / float(bb1_area + bb2_area - intersection_area)

    def prepare_images_and_labels(self, number_of_results=2500, iou_threshold=0.85, max_samples=15, verbose=1):
        '''
        Process rectangles for the images in the DataFrame.
        Generates these rectangles for the learning.
        :param number_of_results: How many rectangles should be processed.
        :param iou_threshold: Precision for the IOU test - 1.0 is a perfect match.
        :param max_samples: How many samples should be kept per class.
        :param verbose: 0:=no output, 1:=errors only, 2:= everything.
        '''
        # Uses optimized selective search
        # https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        # Images and labels
        test_images = []
        test_labels = []

        # Loop over each image
        for index, row in tqdm(self.files.iterrows(), total=self.DataFrame.get_num_files(),
                               desc="Iterating through files...", disable=(not verbose)):
            current_file = row[0]
            data = self.df.loc[self.df['file_name'] == current_file]
            img = cv2.imread(current_file)

            # Set image as the base for selective search
            ss.setBaseImage(img)

            # Initialising fast selective search and getting proposed regions
            ss.switchToSelectiveSearchFast()
            if verbose == 2:
                print("Selective search in progress for file {}.".format(index))
            rects = ss.process()

            img_out = img.copy()

            # We need an uniform sample between classes
            classes_counter = defaultdict(int)

            # Iterate over the first N results of selective search
            # Calculate IOU of proposed region and annoted region
            used = False  # Check if that bbox is used as a class example

            # For each rectangle in the results of selective search
            for i, rect in enumerate(tqdm(rects, desc="Iterating through rectangles...", leave=False,
                                          disable=(verbose != 2))):
                if i < number_of_results:  # We don't want to waste ressources on too many possibilities.
                    x, y, w, h = rect
                    rect_bbox = {'xmin': x, 'xmax': x + w, 'ymin': y, 'ymax': y + h}

                    # For each bbox within the image
                    for index, row in data.iterrows():
                        ground_truth_bbox = {'xmin': int(row['xmin']), 'xmax': int(row['xmax']),
                                             'ymin': int(row['ymin']), 'ymax': int(row['ymax'])}
                        ground_truth_class_name = row['class_name']

                        # Compare them
                        iou = self.get_iou(ground_truth_bbox, rect_bbox)

                        if iou > iou_threshold and classes_counter[ground_truth_class_name] < max_samples:
                            # Get the sample
                            img_sample = cv2.resize(img_out[y:y + h, x:x + w], (224, 224),
                                                    interpolation=cv2.INTER_AREA)
                            test_images.append(img_sample)  # Check if this shit does not bug...
                            test_labels.append(ground_truth_class_name)  # Check if this shit does not bug...
                            classes_counter[ground_truth_class_name] += 1
                            used = True
                        else:
                            continue

                    if not used and classes_counter['background'] < max_samples:
                        # We can use that bbox as a background example!
                        img_sample = cv2.resize(img_out[y:y + h, x:x + w], (224, 224),
                                                interpolation=cv2.INTER_AREA)  # Get the sample
                        test_images.append(img_sample)
                        test_labels.append('background')
                        classes_counter['background'] += 1
                else:
                    break

        self.images = np.array(test_images)
        self.labels = np.array(test_labels)
        self.classes = list(set(self.labels))

    def get_num_samples(self):
        '''
        :return: Number of samples.
        '''
        return self.images.shape[0]

    def get_num_classes(self):
        '''
        :return: Number of classes.
        '''
        return len(self.classes)

    def summary(self):
        '''
        Print a nice little summary of the ImagesData.
        '''
        print("We have {} samples for {} classes.".format(self.get_num_samples(), self.get_num_classes()))
        print(self.classes)

    def show_image(self, id, show_infos=False, show_labels=False):
        '''
        Show an image of the set along with its informations.
        :param id: ID of the image in the dataset.
        :param show_infos: If True, will show informations about the image in the dataset.
        :param show_labels: If True, will show the annotated image along with the "naked" one.
        '''
        sample_row = self.files.iloc[id]
        sample_file = sample_row['file_name']
        sample_data = self.df.loc[self.df['file_name'] == sample_file]

        if show_infos:
            print(sample_data)

        img = cv2.imread(sample_file)
        plt.figure(figsize=(18, 16))
        if show_labels:
            plt.subplot(1, 2, 1)
        plt.imshow(img)

        if show_labels:
            for index, row in sample_data.iterrows():
                class_name, xmin, ymin, xmax, ymax = row['class_name'], int(row['xmin']), int(row['ymin']), int(
                    row['xmax']), int(row['ymax'])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(img, class_name, (xmin + 5, ymin + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
            plt.subplot(1, 2, 2)
            plt.imshow(img)

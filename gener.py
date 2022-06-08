#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
import os
import random
import torch
import numpy as np
import pandas as pd
import glob
from PIL import Image
import itertools
import nvidia.dali.fn as fn
import cupy
import ast


# # External Source Example
#

class MultiClassIterator(object):
    def __init__(self, datasets, batch_size):
        len_df = [len(df) for df in datasets]
        self.largest_dataset = max(len_df) #length of csv file with the most number of datapoints
        self.largest_dataset_idx = len_df.index(self.largest_dataset) #identifier for which one the largest is
        self.reset()
        self.batch_size = batch_size
        self.datasets = datasets
        self.iter_size = sum(len_df)

    def customroundrobin(self, iterable, index):
        '''
        Performs a round robin over the smaller dataset. In the case where one CSV is smaller than the other, the
        smaller one is iterated through again till we reach the length of the largest CSV dataset.
        '''
        start_over = 0
        if index >= len(iterable):
            start_over += 1
        while True:
            for i, element in enumerate(iterable):
                if i >= index or start_over:
                    if i == len(iterable) - 1:
                        start_over += 1
                    yield element

    def __iter__(self):
        return self

    def reset(self):
        self.counter_index = {i: 0 for i, dataset in enumerate(datasets)} #a counter to see how many datapoints have been taken from each dataset
        self.counter_dataset = 0 #identifier for which dataset we're currently on
        self.iterable_datasets = itertools.cycle(datasets)

    def __next__(self):
        batch_crops = []
        batch_labels = [np.array(i) for i in range(4)] #Ignore this, just a dummy label being generated.
        print('Counter Index Updated', self.counter_index)

        #takes a row from our CSV file dataset, and appends the result values to a list. Value is yielded.
        batch_counter = 0
        cur_iterable = self.customroundrobin(self.iterable_datasets.__next__(), self.counter_index[self.counter_dataset])
        while batch_counter < self.batch_size:
            self.counter_index[self.counter_dataset] += 1
            data_point = cur_iterable.__next__()
            crops = data_point["labels_crops"]
            crops = ast.literal_eval(crops)
            crops = [crops[0], crops[1], crops[2] - crops[0], crops[3] - crops[1]] #Format to support ops.slice
            batch_crops.append(np.array([crops], dtype=np.int32))
            batch_counter += 1
        self.counter_dataset += 1
        if self.counter_dataset == len(datasets):
            self.counter_dataset = 0
        if sum(self.counter_index.values()) >= self.size:
            self.reset()
        print(batch_crops, batch_labels)
        return (batch_crops, batch_labels)


    @property
    def size(self):
        return self.largest_dataset


class GenericIterator(DALIGenericIterator):
    def __init__(self, **args):
        super().__init__(**args)
        pass

    def custom_collate(self, loader_dict):
        #using this function for adding custom collate functionalit
        pass

    def __next__(self):
        #Removed a few nuances, but we do require a custom DALIGenericIterator (for custom functions)
        loader_dict = {}
        out = super().__next__()
        out = out[0]
        loader_dict["input"] = out[self.output_map[0]].float()
        loader_dict["labels"] = torch.squeeze(out[self.output_map[1]])

        return loader_dict

datasets = [pd.read_csv(df, index_col= 'Unnamed: 0').to_dict(orient='records') for df in glob.glob('*.csv')]


multi_iter = MultiClassIterator(datasets, 4)


pipe = Pipeline(batch_size=4, num_threads=2, device_id=0)
with pipe:
    jpegs, _ = fn.readers.file(file_list= 'single_image.txt', pad_last_batch=True)
    crop, labels = fn.external_source(multi_iter, num_outputs=2)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    anchor =  fn.reshape(fn.slice(crop, 0, 2, axes=[1]), shape=[-1])
    shape = fn.reshape(fn.slice(crop, 2, 2, axes = [1]), shape= [-1])
    anchor = fn.cast(anchor, dtype=types.INT32)
    shape = fn.cast(shape, dtype=types.INT32)
    images = fn.resize(images, resize_x=224, resize_y=224)
    pipe.set_outputs(images, labels, crop)

pii = GenericIterator(pipelines = pipe, output_map=['data', 'label', 'crops'], auto_reset = True, size = multi_iter.size, last_batch_policy=LastBatchPolicy.PARTIAL)

for i, x in enumerate(pii):
    print(x['labels'], i)
for i, x in enumerate(pii):
    print(x['labels'], i)
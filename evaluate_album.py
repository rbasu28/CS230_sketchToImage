import argparse

import time
import datetime
import pytz 
import os
from PIL import Image

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score
import torch 
import torch.nn as nn

from model.net import BasicModel 
from model.dataloader import AlbumDataloaders, AlbumTestDataset
from utils import *


def evaluate(batch_size, dataloader_fn, images_model, sketches_model, k = 5, num_display = 2):
  device = get_device()
  print(f'Evaluate on {device}')
  images_model, sketches_model = images_model.to(device), sketches_model.to(device)
  images_model.eval()
  sketches_model.eval()

  images_dataloader = dataloader_fn(batch_size = batch_size, section = 'photos', shuffle = False)
  sketches_dataloader = dataloader_fn(batch_size = batch_size, section = 'sketches', shuffle = False)

  '''IMAGES'''
  print('Processing the images. Batch size: %d; Number of batches: %d' % (batch_size, len(images_dataloader)))

  start_time = time.time()

  image_feature_predictions, test_images, test_image_ids = [], [], []
  with torch.no_grad():
    for iteration, batch in enumerate(images_dataloader):
      images, file_ids = batch
      images = torch.autograd.Variable(images.to(device))
      pred_features = images_model(images)
      # Move results to CPU before appending
      test_images.append(images.cpu())
      image_feature_predictions.append(pred_features.cpu())
      test_image_ids.append(file_ids.cpu())
      # image_label_indices.append(label_indices.cpu())

  image_feature_predictions = torch.cat(image_feature_predictions,dim=0)
  test_image_ids = torch.cat(test_image_ids,dim=0)
  test_images = torch.cat(test_images, dim = 0)

  end_time = time.time()

  print('Processed the images. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))


  '''SKETCHES'''
  print('Processing the sketches. Batch size: %d; Number of batches: %d' % (batch_size, len(sketches_dataloader)))

  start_time = time.time()

  sketch_feature_predictions, test_sketches, test_sketch_ids = [], [], []
  with torch.no_grad():
    for iteration, batch in enumerate(sketches_dataloader):
      sketches, sketches_ids = batch
      sketches = torch.autograd.Variable(sketches.to(device))
      pred_features = sketches_model(sketches)
      # Move results to CPU before appending
      test_sketches.append(sketches.cpu())
      sketch_feature_predictions.append(pred_features.cpu())
      test_sketch_ids.append(sketches_ids.cpu())

  sketch_feature_predictions = torch.cat(sketch_feature_predictions,dim=0)
  test_sketch_ids = torch.cat(test_sketch_ids, dim=0)
  test_sketches = torch.cat(test_sketches, dim = 0)

  end_time = time.time()

  print('Processed the sketches. Time taken: %s' % (str(datetime.timedelta(seconds = int(end_time - start_time)))))

  '''mAP calculation'''
  image_feature_predictions = image_feature_predictions.cpu().numpy() 
  sketch_feature_predictions = sketch_feature_predictions.cpu().numpy() 
  test_sketch_ids = test_sketch_ids.cpu().numpy()
  test_image_ids = test_image_ids.cpu().numpy()

  distance = cdist(sketch_feature_predictions, image_feature_predictions, 'minkowski')
  similarity = 1.0/distance 

  # is_correct_label_index = 1 * (np.expand_dims(sketch_label_indices, axis = 1) == np.expand_dims(image_label_indices, axis = 0))

  average_precision_scores = []
  for i in range(sketch_feature_predictions.shape[0]):
    hit_target = is_target_in_list(test_sketch_ids[i], test_image_ids, similarity[i], k)
    average_precision_scores.append(average_precision_score(hit_target, similarity[i]))
  average_precision_scores = np.array(average_precision_scores)

  # index2label = {v: k for k, v in label2index.items()}
  # for cls in set(sketch_label_indices):
  #   print('Class: %s, mAP: %f' % (index2label[cls], average_precision_scores[sketch_label_indices == cls].mean()))

  mean_average_precision = average_precision_scores.mean()

  sketches, image_grids = get_sketch_images_grids(test_sketches, test_images, similarity, k, num_display)

  return sketches, image_grids, mean_average_precision

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluation of SBIR with Album dataset')
  parser.add_argument('--model', help='Trained model path')
  parser.add_argument('--test_photos_dir', help='Album photos directory', required=True)
  parser.add_argument('--test_sketches_dir', help='Album sketches directory', required=True)
  parser.add_argument('--num_images', type=int, help='Number of random images to output for every sketch', default = 5)
  parser.add_argument('--num_sketches', type=int, help='Number of random sketches to output', default = 20)
  parser.add_argument('--batch_size', type=int, help='Batch size to process the test sketches/photos', default = 1)
  parser.add_argument('--output_dir', help='Directory to save output sketch and images', default = 'outputs')

  args = parser.parse_args()

  device = get_device()

  dataloaders = AlbumDataloaders(args.test_photos_dir, args.test_sketches_dir)
  image_model = BasicModel().to(device)
  sketch_model = BasicModel().to(device) 
  if args.model:
    load_checkpoint(args.model, image_model, sketch_model)
  sketches, image_grids, test_mAP = evaluate(args.batch_size, dataloaders.get_test_dataloader, image_model, sketch_model, k = args.num_images, num_display = args.num_sketches)
  print('Average test mAP: ', test_mAP)

  if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

  for i in range(len(sketches)):
    Image.fromarray(np.uint8(sketches[i] * 255)).save(os.path.join(args.output_dir,'Sketch_%d.png'%(i)))
    Image.fromarray(np.uint8(image_grids[i] * 255)).save(os.path.join(args.output_dir,'Images_%d.png'%(i)))

  empty_cache()

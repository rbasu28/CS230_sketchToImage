import os.path
import time
import datetime
import pytz
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

import utils
from model.net import BasicModel, DomainAdversarialNet
from model.dataloader import Dataloaders
from model.layers import grad_reverse
from evaluate import evaluate
from utils import *

class Trainer():
  def __init__(self, data_dir, train_labels, train_embedding, test_labels):
    self.dataloaders = Dataloaders(data_dir, train_labels, train_embedding, test_labels)
    self.train_dict = self.dataloaders.train_dict
    self.test_dict = self.dataloaders.test_dict

  def train_and_evaluate(self, config, checkpoint=None):
    batch_size = config['batch_size']
    # Support both cuda and apple M1/M2 GPU
    device = get_device()
    print(f'Train on {device}')

    train_dataloader = self.dataloaders.get_train_dataloader(batch_size = batch_size, shuffle=True)
    num_batches = len(train_dataloader)

    image_model = BasicModel().to(device)
    sketch_model = BasicModel().to(device)

    domain_net = DomainAdversarialNet().to(device)

    params = [param for param in image_model.parameters() if param.requires_grad == True]
    params.extend([param for param in sketch_model.parameters() if param.requires_grad == True])
    params.extend([param for param in domain_net.parameters() if param.requires_grad == True])
    optimizer = torch.optim.Adam(params, lr=config['lr'])

    criterion = nn.TripletMarginLoss(margin = 1.0, p = 2)
    domain_criterion = nn.BCELoss()
    done_iteration, done_epoch = 0, 0
    if checkpoint:
      checkpoint_path = os.path.join(config['checkpoint_dir'], get_last_pth_file())
      done_iteration, done_epoch = load_checkpoint(checkpoint_path, image_model, sketch_model, domain_net, optimizer)

    print('Training...')

    for epoch in range(config['epochs']):
      accumulated_triplet_loss = RunningAverage()
      accumulated_iteration_time = RunningAverage()
      accumulated_image_domain_loss = RunningAverage()
      accumulated_sketch_domain_loss = RunningAverage()

      epoch_start_time = time.time()

      image_model.train()
      sketch_model.train()
      domain_net.train()

      for iteration, batch in enumerate(train_dataloader):
        time_start = time.time()

        '''GETTING THE DATA'''
        anchors, positives, negatives, label_embeddings, positive_label_idxs, negative_label_idxs = batch
        anchors = torch.autograd.Variable(anchors.to(device)); positives = torch.autograd.Variable(positives.to(device))
        negatives = torch.autograd.Variable(negatives.to(device)); label_embeddings = torch.autograd.Variable(label_embeddings.to(device))

        '''MAIN NET INFERENCE AND LOSS'''
        pred_sketch_features = sketch_model(anchors)
        pred_positives_features = image_model(positives)
        pred_negatives_features = image_model(negatives)

        triplet_loss = config['triplet_loss_ratio'] * criterion(pred_sketch_features, pred_positives_features, pred_negatives_features)
        accumulated_triplet_loss.update(triplet_loss, anchors.shape[0])

        '''DOMAIN ADVERSARIAL TRAINING''' # vannila generator for now. Later - add randomness in outputs of generator, or lower the label

        '''DEFINE TARGETS'''

        image_domain_targets = torch.full((anchors.shape[0],1), 1, dtype=torch.float, device=device)
        sketch_domain_targets = torch.full((anchors.shape[0],1), 0, dtype=torch.float, device=device)

        '''GET DOMAIN NET PREDICTIONS FOR INPUTS WITH G.R.L.'''
        if epoch < 5:
          grl_weight = 0
        elif epoch < config['grl_threshold_epoch']:
          grl_weight *= epoch/config['grl_threshold_epoch']
        else:
          grl_weight = 1

        domain_pred_p_images = domain_net(grad_reverse(pred_positives_features, grl_weight))
        domain_pred_n_images = domain_net(grad_reverse(pred_negatives_features, grl_weight))
        domain_pred_sketches = domain_net(grad_reverse(pred_sketch_features, grl_weight))

        '''DOMAIN LOSS'''

        domain_loss_images = config['domain_loss_ratio'] * (domain_criterion(domain_pred_p_images, image_domain_targets) + domain_criterion(domain_pred_n_images, image_domain_targets))
        accumulated_image_domain_loss.update(domain_loss_images, anchors.shape[0])
        domain_loss_sketches = config['domain_loss_ratio'] * (domain_criterion(domain_pred_sketches, sketch_domain_targets))
        accumulated_sketch_domain_loss.update(domain_loss_sketches, anchors.shape[0])
        total_domain_loss = domain_loss_images + domain_loss_sketches

        '''OPTIMIZATION W.R.T. BOTH LOSSES'''
        optimizer.zero_grad()
        total_loss = triplet_loss + total_domain_loss
        total_loss.backward()
        optimizer.step()


        '''LOGGER'''
        time_end = time.time()
        accumulated_iteration_time.update(time_end - time_start)

        if iteration % config['print_every'] == 0:
          eta_cur_epoch = str(datetime.timedelta(seconds = int(accumulated_iteration_time() * (num_batches - iteration))))
          print(datetime.datetime.now(pytz.timezone('America/Los_Angeles')).replace(microsecond = 0), end = ' ')

          print('Epoch: %d [%d / %d] ; eta: %s' % (epoch, iteration, num_batches, eta_cur_epoch))
          print('Average Triplet loss: %f(%f);' % (triplet_loss, accumulated_triplet_loss()))
          print('Sketch domain loss: %f; Image Domain loss: %f' % (accumulated_sketch_domain_loss(), accumulated_image_domain_loss()))

      '''END OF EPOCH'''
      epoch_end_time = time.time()
      print('Epoch %d complete, time taken: %s' % (epoch, str(datetime.timedelta(seconds = int(epoch_end_time - epoch_start_time)))))
      empty_cache()

      # Estimate done epoch if it is not recorded.
      done_epoch = done_iteration // num_batches if done_epoch == -1 else done_epoch

      save_checkpoint({'iteration': done_iteration + epoch * num_batches,
                        'epoch': done_epoch + epoch,
                        'image_model': image_model.state_dict(),
                        'sketch_model': sketch_model.state_dict(),
                        'domain_model': domain_net.state_dict(),
                        'optim_dict': optimizer.state_dict()},
                      checkpoint_dir=config['checkpoint_dir'])
      print('Saved epoch!')
      print('\n\n\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Training of SBIR')
  parser.add_argument('--data_dir',
                      help='Data directory path. Directory should contain two folders - sketches and photos, along with 2 .txt files for the labels',
                      required=False,
                      default="Dataset/")
  parser.add_argument('--train_labels', help='train label file name', required=False, default='train_labels.txt')
  parser.add_argument('--train_embedding', help='train embedding file name', required=False, default='test_embeddings.npy')
  parser.add_argument('--test_labels', help='test label file name', required=False, default='test_labels.txt')
  parser.add_argument('--batch_size', type=int, help='Batch size to process the train sketches/photos', required=False, default=32)
  parser.add_argument('--checkpoint_dir', help='Directory to save checkpoints', required=False, default='saveModel')
  parser.add_argument('--disable_checkpoint',
                      help='Whether to disable loading checkpoint from previous training',
                      action='store_true')
  parser.add_argument('--epochs', help='Number of epochs', required=False, type=int, default=10)
  parser.add_argument('--lr', help='Learning_rate', required=False, type=float, default=0.001)

  parser.add_argument('--domain_loss_ratio', help='Domain loss weight', required=False, type=float, default=0.5)
  parser.add_argument('--triplet_loss_ratio', help='Triplet loss weight', required=False, type=float, default=1.0)
  parser.add_argument('--grl_threshold_epoch', help='Threshold epoch for GRL lambda', required=False, type=int, default=25)
  parser.add_argument('--print_every', help='Logging interval in iterations', required=False, type=int, default=10)

  args = parser.parse_args()
  # Load from checkpoint if it is not disabled.
  checkpoint = True if os.path.exists(os.path.join(args.checkpoint_dir, get_last_pth_file())) and not args.disable_checkpoint else False
  trainer = Trainer(args.data_dir, args.train_labels, args.train_embedding, args.test_labels)
  trainer.train_and_evaluate(vars(args), checkpoint)

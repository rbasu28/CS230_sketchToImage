import numpy as np
import os
import torch
from torchvision.utils import make_grid

class RunningAverage():
  def __init__(self):
    self.count = 0
    self.sum = 0

  def update(self, value, n_items = 1):
    self.sum += value * n_items
    self.count += n_items

  def __call__(self):
    return self.sum/self.count  


def get_last_pth_file():
    return 'last.pth.tar'

def get_device():
    if torch.cuda.is_available():
      return "cuda"
    elif torch.backends.mps.is_available():
      return "mps"
    else:
      return "cpu"


def empty_cache():
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
      torch.mps.empty_cache()


def save_checkpoint(state, checkpoint_dir):
    file_name = get_last_pth_file()
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, file_name))

def load_checkpoint(checkpoint_path, image_model, sketch_model, domain_model=None, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise Exception("File {} doesn't exist".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(f"Loading the models from the end of net iteration: {checkpoint['iteration']}, epoch: {checkpoint.get('epoch', -1)} (-1 means not saved)")
    image_model.load_state_dict(checkpoint['image_model'])
    sketch_model.load_state_dict(checkpoint['sketch_model'])
    if domain_model:
        domain_model.load_state_dict(checkpoint['domain_model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint['iteration'], checkpoint.get("epoch", -1)

def get_sketch_images_grids(sketches, images, similarity_scores, k, num_display):

  if num_display == 0 or k == 0:
    return None, None
  num_sketches = sketches.shape[0]
  np.random.seed(234)  # used to display consistent samples output.
  if num_sketches < num_display:
    num_display = num_sketches
    indices = np.arange(num_sketches)
  else:
    indices = np.random.choice(num_sketches, num_display)

  cur_sketches = sketches[indices]; cur_similarities = similarity_scores[indices]
  top_k_similarity_indices  = np.flip(np.argsort(cur_similarities, axis = 1)[:, -k:], axis = 1).copy()
  top_k_similarity_values = np.flip(np.sort(cur_similarities, axis = 1)[:,-k:], axis = 1).copy()
  matched_images = [images[top_k_similarity_indices[i]] for i in range(num_display)]

  list_of_sketches = [np.transpose(cur_sketches[i].cpu().numpy(), (1,2,0)) for i in range(num_display)]
  list_of_image_grids = [np.transpose(make_grid(matched_images[i], nrow = k).cpu().numpy(), (1,2,0)) for i in range(num_display)]

  return list_of_sketches, list_of_image_grids


def is_target_in_list(sketch_id, test_image_ids, similarity_scores, k):
    # todo, implement
    indices = np.argsort(similarity_scores)
    k_indices = indices[-k:]
    k_flip = np.flip(k_indices)
    top_k_similarity_indices = k_flip.copy()
    result_image_indices = test_image_ids[k_flip]
    result = np.zeros_like(similarity_scores)
    result[k_flip] = 1
    return result
    # top_k_similarity_indices = np.flip(np.argsort(similarity_scores, axis=1)[:, -k:], axis=1).copy()


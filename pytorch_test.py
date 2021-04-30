import numpy as np
import cv2
from pytorch_warp import warp_imgs

import torch as pt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

def load_image(image_path):
  img = cv2.imread(image_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return pt.from_numpy(img)

if __name__ == '__main__':
  dev = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')
  print(f'Using {dev} device')

  # prepare data
  class WarpDataSet(Dataset):
    def __init__(self, pixel_coords_targ, k_s, k_t, rot, t, n_trans, a, img_targ):
      self.data_list = [(pixel_coords_targ, k_s, k_t, rot, t, n_trans, a)]
      self.target = [img_targ]

    def __getitem__(self, index):
      return (self.data_list[index], self.target[index])

    def __len__(self):
      return len(self.data_list)
  class WrappedDataLoader:
    def __init__(self, dl, func):
      self.dl = dl
      self.func = func

    def __len__(self):
      return len(self.dl)

    def __iter__(self):
      batches = iter(self.dl)
      for b in batches:
        yield (self.func(*b))
  def preprocess(x, y):
    return tuple([tuple(x[i].to(dev) for i in range(len(x))), y.to(dev)])

  # right-hand opencv coordinate system
  R_src = pt.tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=pt.float32)
  T_src = pt.unsqueeze(pt.tensor([0, 0, 0], dtype=pt.float32), -1)
  value = 2 ** 0.5 * 0.5
  R_targ = pt.tensor([[value,value,0], [-value,value,0], [0,0,1]], dtype=pt.float32)
  T_targ = pt.unsqueeze(pt.tensor([0, 0, 0], dtype=pt.float32), -1)
  K_src = pt.tensor([[2400,0,959.5], [0,2400,539.5], [0,0,1]], dtype=pt.float32)
  K_targ = pt.tensor([[3200,0,959.5], [0,3200,539.5], [0,0,1]], dtype=pt.float32)

  img_gt = load_image('data/src.jpg').to(pt.float32) / 255
  im_h, im_w = img_gt.shape[-3:-1]
  ys, xs = pt.meshgrid(pt.arange(im_h), pt.arange(im_w))
  pixel_coords_targ = pt.stack([xs, ys], -1)
  k_s = K_src
  k_t = K_targ
  rot = rot_src_to_targ = R_targ.t() @ R_src
  t = t_src_to_targ = R_targ.t() @ (T_src - T_targ)
  n_trans = pt.tensor([[0,0,1]], dtype=pt.float32)
  a = pt.tensor([[2.5]])
  img_targ = load_image('data/target.jpg').to(pt.float32) / 255

  # create model
  class WarpModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.src = pt.randn_like(img_gt)
      self.src = nn.Parameter(pt.stack([self.src], 0))
      self.warp_func = warp_imgs

      self.mask = None

    def forward(self, xb):
      pixel_coords_targ = xb[0]
      k_s = xb[1]
      k_t = xb[2]
      rot = xb[3]
      t = xb[4]
      n_trans = xb[5]
      a = xb[6]
      warped_result = self.warp_func(
        self.src, pixel_coords_targ, k_s, k_t, rot, t, n_trans, a)

      if self.mask is None:
        ones = pt.ones_like(self.src)
        self.mask = self.warp_func(
          ones, pixel_coords_targ, k_s, k_t, rot, t, n_trans, a) != 0

      return warped_result

  # train model parameters
  def loss_func(input, target, mask):
    return pt.mean(pt.square(input - target)[mask])
  def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb, model.mask)

    if opt is not None:
      loss.backward()
      opt.step()
      opt.zero_grad()

    return loss.item(), len(xb)
  def fit(epochs, model, loss_func, opt, train_dl, test_dl=None):
    for epoch in range(epochs):
      model.train()
      for xb, yb in train_dl:
        loss, _ = loss_batch(model, loss_func, xb, yb, opt)
      print(epoch, loss)

      if test_dl is not None:
        model.eval()
        with pt.no_grad():
          losses, nums = zip(
            *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl]
          )
        valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, valid_loss)




  batch_size = 1
  lr = 1e3
  momentum = 0.9
  epochs = 1000

  train_ds = WarpDataSet(
    pixel_coords_targ, k_s, k_t, rot, t, n_trans, a, img_targ)
  train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
  train_dl = WrappedDataLoader(train_dl, preprocess)
  model = WarpModel().to(dev)
  opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  fit(epochs, model, loss_func, opt, train_dl)

  trained_result = pt.squeeze(model.src.cpu().detach()).numpy()
  from matplotlib import pyplot as plt
  plt.figure()
  plt.get_current_fig_manager().window.showMaximized()
  plt.imshow(img_gt)
  plt.tight_layout()
  plt.figure()
  plt.get_current_fig_manager().window.showMaximized()
  plt.imshow(trained_result)
  plt.tight_layout()
  plt.show()

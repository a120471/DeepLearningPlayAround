# modified version of https://github.com/google/stereo-magnification/blob/master/geometry/homography.py

import torch as pt

def divide_safe(num, den):
  eps = 1e-8
  den += eps * pt.eq(den, 0).to(pt.float32)
  return pt.div(num, den)

def inv_homography(k_s, k_t, rot, t, n_trans, a):
  '''Computes inverse homography matrix between two cameras via a plane.

  Args:
    k_s: intrinsics for source cameras, [..., 3, 3] matrices
    k_t: intrinsics for target cameras, [..., 3, 3] matrices
    rot: relative rotations between source and target, [..., 3, 3] matrices
    t: [..., 3, 1], translations from source to target camera. Mapping a 3D
      point p from source to target is accomplished via rot * p + t
    n_trans: [..., 1, 3], plane normal w.r.t source camera frame
    a: [..., 1, 1], plane equation displacement
  Returns:
    homography: [..., 3, 3] inverse homography matrices (homographies mapping
      pixel coordinates from target to source)
  '''
  rot_t = rot.transpose(-1, -2)
  k_t_inv = k_t.inverse()

  denom = a - n_trans @ rot_t @ t
  numerator = rot_t @ t @ n_trans @ rot_t
  inv_hom = k_s @ (rot_t + divide_safe(numerator, denom)) @ k_t_inv
  return inv_hom

def transform_points(points, homography):
  '''Transforms input points according to homography.

  Args:
    points: [..., H, W, 2]; pixel (u,v) coordinates
    homography: [..., 3, 3]; desired matrix transformation
  Returns:
    output_points: [..., H, W, 2]; transformed (u,v) coordinates
  '''
  points = points.to(pt.float32)
  ones = pt.ones_like(points[..., 0:1])
  points = pt.cat([points, ones], -1)
  points_reshaped = points.reshape(
    [-1, points.shape[-3]*points.shape[-2], points.shape[-1]])
  transformed_points = points_reshaped @ homography.transpose(-1, -2)
  transformed_points = transformed_points.reshape(points.shape)
  uv = transformed_points[..., :-1]
  w = transformed_points[..., -1:]
  return divide_safe(uv, w)

def bilinear_wrapper(imgs, coords):
  '''Wrapper around bilinear sampling function, handles arbitrary input sizes.

  Args:
    imgs: [..., H_s, W_s, C] images to resample
    coords: [..., H_t, W_t, 2], source pixel locations from which to copy
  Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input
  '''
  # convert coords to [-1, 1]
  im_h, im_w = imgs.shape[-3:-1]
  coords[..., 0] = coords[..., 0] / (im_w-1) * 2 - 1
  coords[..., 1] = coords[..., 1] / (im_h-1) * 2 - 1

  # The bilinear sampling code only handles 4D input, so we'll need to reshape.
  init_dims = imgs.shape[:-3]
  end_dims_img = imgs.shape[-3:]
  end_dims_coords = coords.shape[-3:]
  prod_init_dims = init_dims.numel()

  imgs = imgs.reshape([prod_init_dims] + list(end_dims_img))
  imgs = imgs.to(pt.float32)
  imgs = pt.einsum('...ijk->...kij', imgs)
  coords = coords.reshape([prod_init_dims] + list(end_dims_coords))
  imgs_sampled = pt.nn.functional.grid_sample(imgs, coords, align_corners=True)
  imgs_sampled = pt.einsum('...kij->...ijk', imgs_sampled)
  return imgs_sampled.reshape(init_dims + imgs_sampled.shape[-3::])

def warp_imgs(imgs, pixel_coords_targ, k_s, k_t, rot, t, n_trans, a):
  '''Transforms input imgs via homographies for corresponding planes.

  Args:
    imgs: [..., H_s, W_s, C]
    pixel_coords_targ: [..., H_t, W_t, 2]; pixel (u,v) coordinates
    k_s: intrinsics for source cameras, [..., 3, 3] matrices
    k_t: intrinsics for target cameras, [..., 3, 3] matrices
    rot: relative rotation, [..., 3, 3] matrices
    t: [..., 3, 1], translations from source to target camera
      point p from source to target is accomplished via rot * p + t
    n_trans: [..., 1, 3], plane normal w.r.t source camera frame
    a: [..., 1, 1], plane equation displacement
  Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
      Coordinates outside the image are sampled as 0
  '''
  hom_t2s = inv_homography(k_s, k_t, rot, t, n_trans, a)
  pixel_coords_t2s = transform_points(pixel_coords_targ, hom_t2s)
  imgs_s2t = bilinear_wrapper(imgs, pixel_coords_t2s)

  return imgs_s2t

# modified version of https://github.com/google/stereo-magnification/blob/master/geometry/homography.py

import tensorflow as tf
import tensorflow_addons as tfa

def divide_safe(num, den, name=None):
  eps = 1e-8
  den += eps * tf.cast(tf.equal(den, 0), tf.float32)
  return tf.divide(num, den, name=name)

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
  with tf.name_scope('inv_homography'):
    rot_t = tf.linalg.matrix_transpose(rot)
    k_t_inv = tf.linalg.inv(k_t, name='k_t_inv')

    denom = a - n_trans @ rot_t @ t
    numerator = rot_t @ t @ n_trans @ rot_t
    inv_hom = k_s @ (rot_t + divide_safe(numerator, denom)) @ k_t_inv
    return inv_hom

def transform_points(points, homography):
  '''Transforms input points according to homography.

  Args:
    points: [..., H, W, 2]; pixel (u,v,1) coordinates
    homography: [..., 3, 3]; desired matrix transformation
  Returns:
    output_points: [..., H, W, 2]; transformed (u,v) coordinates
  '''
  with tf.name_scope('transform_points'):
    points = tf.cast(points, tf.float32)
    ones = tf.ones_like(points[..., 0:1])
    points = tf.concat([points, ones], -1)
    points_reshaped = tf.reshape(points,
      [-1, points.shape[-3]*points.shape[-2], points.shape[-1]])
    transformed_points = tf.matmul(points_reshaped, homography, transpose_b=True)
    transformed_points = tf.reshape(transformed_points, points.shape)
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
  # The bilinear sampling code only handles 4D input, so we'll need to reshape.
  init_dims = imgs.shape[:-3]
  end_dims_img = imgs.shape[-3:]
  end_dims_coords = coords.shape[-3:]
  prod_init_dims = tf.math.cumprod(init_dims)[-1]

  imgs = tf.reshape(imgs, [prod_init_dims] + end_dims_img)
  imgs = tf.cast(imgs, tf.float32)
  coords = tf.reshape(coords, [prod_init_dims] + end_dims_coords)
  imgs_sampled = tfa.image.resampler(imgs, coords)
  imgs_sampled = tf.reshape(
    imgs_sampled, init_dims + imgs_sampled.shape[-3::])
  return imgs_sampled

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
  with tf.name_scope('warp_imgs'):
    hom_t2s = inv_homography(k_s, k_t, rot, t, n_trans, a)
    pixel_coords_t2s = transform_points(pixel_coords_targ, hom_t2s)
    imgs_s2t = bilinear_wrapper(imgs, pixel_coords_t2s)

    return imgs_s2t

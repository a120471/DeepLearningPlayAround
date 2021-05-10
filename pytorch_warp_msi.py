import torch as pt
import math

def divide_safe(num, den):
  eps = 1e-8
  den += eps * pt.eq(den, 0).to(pt.float32)
  return pt.div(num, den)

def calculate_rays_dir(points, k_t, rot):
  '''Computes rays direction transformed by rot.

  Args:
    points: [..., H, W, 2]; pixel (u,v) coordinates
    k_t: intrinsics for target cameras, [..., 3, 3] matrices
    rot: relative rotations between target and source, [..., 3, 3] matrices
  Returns:
    rays: [..., H, W, 3] transformed rays direction
  '''
  points = points.to(pt.float32)
  ones = pt.ones_like(points[..., 0:1])
  points = pt.cat([points, ones], -1)
  points_reshaped = points.reshape(
    [-1, points.shape[-3]*points.shape[-2], points.shape[-1]]).transpose(-1, -2)

  k_t_inv = k_t.inverse()

  rays_dir = rot @ k_t_inv @ points_reshaped
  rays_dir = rays_dir.transpose(-1, -2).reshape(points.shape)

  return rays_dir / pt.linalg.norm(rays_dir, axis=-1).unsqueeze(-1)

def calculate_ray_sphere_intersection(rays_o, rays_dir, radius):
  '''Transforms input points according to homography.

  Args:
    rays_o: [..., 3], origin position of rays
    rays_dir: [..., H, W, 3], direction of rays

    radius: [..., 1]; radius of sphere
  Returns:
    intersections: [..., H, W, 3]; intersection points
  '''
  # (rays_o + rays_dir * t - sphere_center) ^ 2 == radius ^ 2
  # At^2 + Bt + C = 0, solve t

  A2 = pt.linalg.norm(rays_dir, axis=-1, keepdim=True) ** 2 * 2.0
  B = 2.0 * pt.sum(rays_dir * rays_o[...,None,None,:], axis=-1, keepdim=True)
  C = pt.linalg.norm(rays_o, axis=-1, keepdim=True) ** 2 - radius ** 2

  det = B ** 2 - 2.0 * A2 * C[...,None,None]
  mask = det > 0.0
  det = pt.abs(det)
  sqrt_det = det ** 0.5
  t_smaller = (-B - sqrt_det) / A2
  t = (-B + sqrt_det) / A2

  t[t_smaller > 0] = t_smaller[t_smaller > 0]
  mask &= t > 0.0
  return rays_o[...,None,None,:] + t * rays_dir, mask[...,0]

# right-hand opencv coordinate system
def convert_point3d_to_theta_phi(points3d):
  theta = pt.atan2(points3d[...,2], -points3d[...,0]) # tan(theta) = z / -x
  c = (points3d[...,2] ** 2 + points3d[...,0] ** 2) ** 0.5
  phi = math.pi / 2 + pt.atan2(points3d[...,1], c) # tan(90 + theta) = y / c
  return pt.stack([theta, phi], dim=-1)

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

  new_imgs_shape = [coords.shape[i] // imgs.shape[i]
    if coords.shape[i] != imgs.shape[i] else -1 for i in range(len(imgs.shape) - 3)]
  new_imgs_shape += [-1, -1, -1]
  imgs = imgs.expand(new_imgs_shape)

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

def warp_imgs_msi(imgs, pixel_coords_targ, k_t, rot, t, r):
  '''Transforms input hemisphere imgs for corresponding planes.

  Args:
    imgs: [..., H_s, W_s, C]
    pixel_coords_targ: [..., H_t, W_t, 2]; pixel (u,v) coordinates
    k_t: intrinsics for target cameras, [..., 3, 3] matrices
    rot: relative rotation, [..., 3, 3] matrices
    t: [..., 3], translations from target to source camera
      point p from target to source is accomplished via rot * p + t
    r: [..., 1], hemisphere radius
  Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
      Coordinates outside the image are sampled as 0
  '''
  rays_dir = calculate_rays_dir(pixel_coords_targ, k_t, rot)
  intersections, mask = calculate_ray_sphere_intersection(t, rays_dir, r)
  theta_phi_coord = convert_point3d_to_theta_phi(intersections) # theta in (0,pi); phi in (0,pi)
  hs, ws = imgs.shape[-3], imgs.shape[-2]
  theta_phi_coord[..., 0] = theta_phi_coord[..., 0] / math.pi * ws + 0.5
  theta_phi_coord[..., 1] = theta_phi_coord[..., 1] / math.pi * hs + 0.5
  imgs_s2t = bilinear_wrapper(imgs, theta_phi_coord)
  imgs_s2t[~mask] = 0

  return imgs_s2t

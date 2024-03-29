import math

import torch as pt
from torch import nn

from pytorch_warp_plane import bilinear_interpolate, divide_safe


def _calculate_rays_dir(points, CF, K):
    """Computes rays direction in world space.

    Args:
      points: [..., H, W, 2]; pixel (u,v) coordinates
      CF: camera frame, [..., 4, 4]
      K: intrinsics for cameras, [..., 3, 3]

    Returns:
      rays: [..., H, W, 3] transformed rays direction
    """
    points = points.to(pt.float32)
    ones = pt.ones_like(points[..., 0:1])
    points = pt.cat([points, ones], -1)
    points_reshaped = points.reshape(
        [-1, points.shape[-3] * points.shape[-2], points.shape[-1]]
    ).transpose(-1, -2)

    R_inv = CF[..., :3, :3]
    K_inv = K.inverse()

    rays_dir = R_inv @ K_inv @ points_reshaped
    rays_dir = rays_dir.transpose(-1, -2).reshape(points.shape)
    rays_dir = nn.functional.normalize(rays_dir, p=2, dim=-1)

    return rays_dir


def _calculate_ray_sphere_intersection(rays_o, rays_dir, sphere_o, sphere_r):
    """Calculates intersection points between rays and sphere.

    Args:
      rays_o: [..., 3], origin position of rays
      rays_dir: [..., H, W, 3], direction of rays
      sphere_o: [..., 3]; center of sphere
      sphere_r: [..., 1]; radius of sphere

    Returns:
      intersections: [..., H, W, 3]; intersection points
      mask: [..., H, W]; valid intersection mask
    """
    # (rays_dir * t + rays_o - sphere_o) ^ 2 == sphere_r ^ 2
    # At^2 + Bt + C = 0, solve t

    o = rays_o - sphere_o
    A = pt.sum(rays_dir**2, axis=-1, keepdim=True)
    B = 2 * pt.sum(rays_dir * o, axis=-1, keepdim=True)
    C = pt.sum(o**2, axis=-1) - sphere_r**2

    det = B**2 - 4 * A * C.squeeze(-1).squeeze(-1)
    sqrt_det = pt.abs(det) ** 0.5
    t1 = divide_safe(-B - sqrt_det, A * 2)
    t2 = divide_safe(-B + sqrt_det, A * 2)

    t2[t1 >= 0] = t1[t1 >= 0]
    mask = det >= 0
    t2[~mask] = 0
    return rays_o.unsqueeze(-2).unsqueeze(-2) + t2 * rays_dir, mask[..., 0]


# in right-hand opencv coordinate system
def _convert_point3d_to_theta_phi(CF1, points):
    """Converts 3D points in world frame to theta_phi coordinates in camera frame.

    Args:
      CF1: camera frame, [..., 4, 4]
      points: [..., H, W, 3]; 3D points in world frame

    Returns:
      theta_phi: [..., H, W, 2]; theta_phi coordinates in camera frame,
        where theta is in [-pi, pi] and phi is in [-pi/2, pi/2]
    """
    R = CF1[..., :3, :3].transpose(-1, -2)
    T = -R @ CF1[..., :3, 3:4]
    points_reshaped = points.reshape(
        [-1, points.shape[-3] * points.shape[-2], points.shape[-1]]
    ).transpose(-1, -2)
    points_reshaped = R @ points_reshaped + T
    points = points_reshaped.transpose(-1, -2).reshape(points.shape)

    # theta = atan(x / z)
    theta = pt.atan2(points[..., 0], points[..., 2])
    c = pt.hypot(points[..., 0], points[..., 2])
    # phi = atan(y / c)
    phi = pt.atan2(points[..., 1], c)
    return pt.stack([theta, phi], dim=-1)


def warp_imgs_to_sphere(imgs, pixel_coords, CF1, CF2, K2, sphere_radius):
    """Warp images from cam1 to cam2. To achieve this, pixel coords are
      un-projected from cam2 first, calculate the ray-sphere intersection,
      get the 2d coords in cam1, and finally bilinear sample the color from imgs.

    Args:
      All data below are in world frame.
      imgs: [..., C, H1, W1]
      pixel_coords: [..., H2, W2, 2]
      CF1: camera frame of cam1, [..., 4, 4]
      CF2: camera frame of cam2, [..., 4, 4]
      K2: intrinsics for cam2, [..., 3, 3]
      sphere_radius: [..., 1], hemisphere radius
    Returns:
      Images after bilinear sampling from imgs. Coordinates outside the image are
      sampled as 0. [..., C, H2, W2]
    """
    rays_dir = _calculate_rays_dir(pixel_coords, CF2, K2)
    intersections, mask = _calculate_ray_sphere_intersection(
        CF2[..., :3, 3], rays_dir, CF1[..., :3, 3], sphere_radius
    )
    theta_phi_coords = _convert_point3d_to_theta_phi(CF1, intersections)
    # imgs represent hemisphere with theta[-pi/2, pi/2], phi[-pi/2, pi/2]
    img_h, img_w = imgs.shape[-2:]
    img_coords = pt.zeros_like(theta_phi_coords)
    img_coords[..., 0] = (theta_phi_coords[..., 0] / math.pi + 0.5) * (img_w - 1)
    img_coords[..., 1] = (theta_phi_coords[..., 1] / math.pi + 0.5) * (img_h - 1)
    imgs_warped = bilinear_interpolate(imgs, img_coords)
    imgs_warped = imgs_warped * mask.unsqueeze(-3)

    return imgs_warped

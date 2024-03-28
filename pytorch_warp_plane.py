# a pytorch version of https://github.com/google/stereo-magnification/blob/master/geometry/homography.py
import torch as pt


def _divide_safe(numerator, denom):
    eps = 1e-8
    denom += eps * pt.eq(denom, 0).to(pt.float32)
    return pt.div(numerator, denom)


def _get_homography(CF1, K1, CF2, K2, plane_param):
    """Computes homography matrix between two cameras via a plane, which maps
      coordinates from cam2 to cam1.

    Ref:
      https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html#tutorial_homography_Demo3

    Args:
      CF1: camera frame of cam1, [..., 4, 4]
      K1: intrinsics for cam1, [..., 3, 3]
      CF2: camera frame of cam2, [..., 4, 4]
      K2: intrinsics for cam2, [..., 3, 3]
      plane_param: params of a plane, [..., 4, 1]

    Returns:
      homography: [..., 3, 3]
    """
    R1 = CF1[..., :3, :3].transpose(-1, -2)
    P1 = CF1[..., :3, 3:4]
    R2_inv = CF2[..., :3, :3]
    P2 = CF2[..., :3, 3:4]
    K2_inv = K2.inverse()
    N_t = plane_param[..., :3, :1].transpose(-1, -2)
    d = (N_t @ P2 + plane_param[..., 3:4, :1]).squeeze(-1)

    H = K1 @ R1 @ (R2_inv - _divide_safe((P2 - P1) @ N_t @ R2_inv, d)) @ K2_inv
    return H


def _transform_points(H, points):
    """Transforms input points according to homography.

    Args:
      H: [..., 3, 3]; desired matrix transformation
      points: [..., H, W, 2]; pixel (u,v) coordinates

    Returns:
      output_points: [..., H, W, 2]; transformed (u,v) coordinates
    """
    points = points.to(pt.float32)
    ones = pt.ones_like(points[..., 0:1])
    points = pt.cat([points, ones], -1)
    points_reshaped = points.reshape(
        [-1, points.shape[-3] * points.shape[-2], points.shape[-1]]
    )
    transformed_points = points_reshaped @ H.transpose(-1, -2)
    transformed_points = transformed_points.reshape(points.shape)
    uv = transformed_points[..., :-1]
    w = transformed_points[..., -1:]
    return _divide_safe(uv, w)


def _bilinear_wrapper(imgs, coords):
    """Wrapper around bilinear sampling function, handles arbitrary input sizes.

    Args:
      imgs: [..., C, H1, W1]
      coords: [..., H2, W2, 2]

    Returns:
      [..., H2, W2, C] images after bilinear sampling from imgs
    """
    # convert coords value to [-1, 1]
    img_h, img_w = imgs.shape[-2:]
    coords[..., 0] = coords[..., 0] / (img_w - 1) * 2 - 1
    coords[..., 1] = coords[..., 1] / (img_h - 1) * 2 - 1

    # The bilinear sampling code only handles 4D input, so we need to reshape.
    init_dims = imgs.shape[:-3]
    prod_init_dims = init_dims.numel()
    end_dims_img = imgs.shape[-3:]
    end_dims_coords = coords.shape[-3:]

    imgs = imgs.reshape([prod_init_dims] + list(end_dims_img))
    imgs = imgs.to(pt.float32)
    coords = coords.reshape([prod_init_dims] + list(end_dims_coords))
    imgs_sampled = pt.nn.functional.grid_sample(imgs, coords, align_corners=True)
    return imgs_sampled.reshape(init_dims + imgs_sampled.shape[-3:])


def warp_imgs(imgs, pixel_coords, CF1, K1, CF2, K2, plane_param):
    """Warp images from cam1 to cam2. To achieve this, pixel coords are
      transformed from cam2 to cam1 using homography matrix. Here we assume that
      the 3D points are on a plane, and the color of the warped pixels are
      bilinear sampled using data in imgs.

    Args:
      All data below are in world frame.
      imgs: [..., C, H1, W1]
      pixel_coords: [..., H2, W2, 2]
      CF1: camera frame of cam1, [..., 4, 4]
      K1: intrinsics for cam1, [..., 3, 3]
      CF2: camera frame of cam2, [..., 4, 4]
      K2: intrinsics for cam2, [..., 3, 3]
      plane_param: params of a plane, [..., 4, 1]

    Returns:
      Images after bilinear sampling from imgs. Coordinates outside the image are
      sampled as 0. [..., C, H2, W2]
    """
    H = _get_homography(CF1, K1, CF2, K2, plane_param)
    transformed_coords = _transform_points(H, pixel_coords)
    imgs_warped = _bilinear_wrapper(imgs, transformed_coords)

    return imgs_warped

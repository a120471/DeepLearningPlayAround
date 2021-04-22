from tensorflow_warp import warp_imgs
import tensorflow as tf

def tf_load_jpeg(image_path):
  data = tf.io.read_file(image_path)
  return tf.io.decode_jpeg(data, dct_method='INTEGER_ACCURATE')

if __name__ == '__main__':
  # prepare test data
  R_src = tf.convert_to_tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=tf.float32)
  T_src = tf.expand_dims(tf.convert_to_tensor([0, 0, -2.5], dtype=tf.float32), -1)
  value = 2**0.5 / 2
  R_targ = tf.convert_to_tensor([[value,value,0], [-value,value,0], [0,0,1]], dtype=tf.float32)
  T_targ = tf.expand_dims(tf.convert_to_tensor([0, 0, -2.5], dtype=tf.float32), -1)
  K_src = tf.convert_to_tensor([[2400,0,959.5], [0,2400,539.5], [0,0,1]], dtype=tf.float32)
  K_targ = tf.convert_to_tensor([[3200,0,959.5], [0,3200,539.5], [0,0,1]], dtype=tf.float32)
  # the above coordinates are in right hand opencv axes


  img_src = tf.cast(tf_load_jpeg('data/src.jpg'), tf.float32) / 255
  img_targ = tf.cast(tf_load_jpeg('data/target.jpg'), tf.float32) / 255
  img_src_random = tf.random.normal(img_src.shape)
  imgs = tf.Variable(tf.stack([img_src_random], 0))

  im_h, im_w = imgs.shape[-3:-1]
  xs, ys = tf.meshgrid(range(im_w), range(im_h))
  pixel_coords_targ = tf.stack([tf.stack([xs, ys], -1)] * 1, 0)

  k_s = tf.stack([K_src] * 1, 0)
  k_t = tf.stack([K_targ] * 1, 0)

  rot_src_targ = tf.matmul(R_targ, R_src, transpose_a=True)
  rot = tf.stack([rot_src_targ], 0)

  t_src_targ = tf.matmul(R_targ, T_src - T_targ, transpose_a=True)
  t = tf.stack([t_src_targ], 0)

  # planes
  # z = [d1, d2...] (in target camera frame)
  n_trans = tf.stack([-rot_src_targ[2:]], 0)
  for d in [2.5]:
    a = tf.stack([-t_src_targ[2, 0] + d], 0)
    a = tf.reshape(a, [-1, 1, 1])

    for i in range(1000):
      with tf.GradientTape() as tape:
        warped_result = warp_imgs(imgs, pixel_coords_targ, k_s, k_t, rot, t, n_trans, a)
        loss = tf.reduce_mean(tf.square(warped_result - img_targ))

      print(i, loss)
      grads = tape.gradient(loss, imgs)
      imgs.assign_sub(grads * 1e5)

    from matplotlib import pyplot as plt
    for i1, i2 in zip(imgs.numpy(), warped_result):
      plt.figure()
      plt.imshow(tf.cast(i1 * 255, tf.uint8))
      plt.figure()
      plt.imshow(tf.cast(i2 * 255, tf.uint8))
      plt.show()

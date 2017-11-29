import numpy as np

def max_pool(inputs, window_dim_1, window_dim_2, stride=1):
    N, C, H, W = inputs.shape
    assert (H - window_dim_1) % stride == 0
    assert (W - window_dim_2) % stride == 0
    H_no = (H - window_dim_1) // stride + 1
    W_no = (W - window_dim_2) // stride + 1
    t3 = np.repeat(np.arange(H_no*W_no, dtype=int), window_dim_2*window_dim_2).\
             reshape(H_no, W_no*window_dim_1*window_dim_2)*stride
    t4 = t3 + (np.arange(W_no)*stride*H_no).reshape(-1,1)

    t5 = np.zeros([window_dim_1,window_dim_2], dtype=int)+np.arange(window_dim_2)
    t6 = (t5 + (np.arange(window_dim_1)*W).reshape(-1,1)).flatten()
    t8 = t4.reshape(-1, window_dim_1*window_dim_2)+t6
    inputs_reshaped= inputs.reshape(N, C, -1)
    output = inputs_reshaped[:,:,t8]
    output = output.reshape(-1, 4)
    weight = output.argmax(axis=1)
    weight = (weight[:, np.newaxis] == np.arange(window_dim_1 * window_dim_2))
    new_weight = np.zeros([N,C,H*W])
    new_weight[:,:,t8] = weight.reshape(N,C,H_no*W_no, window_dim_1*window_dim_2)
    output = output.max(axis=1)
    return [output.reshape(N,C,H_no,W_no), new_weight.reshape(N, C, H, W)]


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) // stride + 1
  out_width = (W + 2 * padding - field_width) // stride + 1

  i0 = np.repeat(np.arange(field_height, dtype=int), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height, dtype=int), out_width)
  j0 = np.tile(np.arange(field_width, dtype=int), field_height * C)
  j1 = stride * np.tile(np.arange(out_width, dtype=int), int(out_height))
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)
  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]

# a = np.arange(20*25).reshape(2,10,5,5)
# k = np.arange(30*4).reshape(3,10,2,2)
# k = k.reshape(3,-1)
# b = im2col_indices(a, 2,2,padding=0)
# out = np.dot(k, b)
# c = col2im_indices(out,(2,3,4,4),4,4,padding=0)
#
# print b.shape
# print k.shape


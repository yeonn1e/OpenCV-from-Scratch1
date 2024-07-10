import numpy as np
import cv2
import time

lenna = cv2.imread('lenna.png' , cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread('shapes.png' , cv2.IMREAD_GRAYSCALE)

# A1-1 functions
def gaussian(i,j,sigma):
  return (1/(2 * np.pi * sigma**2)) * np.exp(-1*((i**2 + j**2)/(2*(sigma**2))))

def get_gaussian_filter_2d(size, sigma):
  half = size // 2
  px, py = np.meshgrid(np.arange(-half,half+1), np.arange(-half,half+1))
  kernel = gaussian(px,py,sigma)
  kernel /= np.sum(kernel)

  return kernel

def padding(img, pad):
  padded_img = np.zeros((img.shape[0]+2*pad,img.shape[1]+2*pad))

  padded_img[pad:-pad, pad:-pad] = img # filtered img 내부는 원이미지대로

  padded_img[:pad, :pad] = img[0, 0] # 모서리 특정 부분 동일한 값으로 설정
  padded_img[-pad:, -pad:] = img[-1, -1]
  padded_img[:pad, -pad:] = img[0, -1]
  padded_img[-pad:, :pad] = img[-1, 0]

  padded_img[:pad, pad:-pad] = img[0,:] # 가장자리 값 img와 동일하게 설정
  padded_img[pad:-pad, :pad] = img[:,0].reshape(-1, 1)
  padded_img[pad:-pad, -pad:] = img[:,-1].reshape(-1, 1)
  padded_img[-pad:, pad:-pad] = img[-1,:]

  return padded_img

def cross_correlation_2d(img, kernel):
  kernelsize = kernel.shape[0]
  pad = kernelsize//2
  filtered_img = np.zeros((img.shape[0], img.shape[1]))

  padded = padding(img, pad)

  for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        convolution = padded[i:i + kernelsize, j:j + kernelsize]
        filtering = (convolution * kernel).sum()
        filtered_img[i, j] = filtering

  return filtered_img

def compute_image_gradient(img):
    start = time.time()
    mag = np.zeros(img.shape)
    dir = np.zeros(img.shape)
    sobelX = np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
    sobelY = sobelX.T # transpose

    operate_x = cross_correlation_2d(img, sobelX)
    operate_y = cross_correlation_2d(img, sobelY)
    mag = np.hypot(operate_x, operate_y)
    dir = np.arctan2(operate_y, operate_x)

    end = time.time() - start

    print('computational time of computing image gradient:', end)

    return mag, dir

filter = get_gaussian_filter_2d(7, 1.5)

# lenna.png
print('++++++mag lenna+++++')
filtered_L = cross_correlation_2d(lenna, filter)
magL, dirL = compute_image_gradient(filtered_L)
# show and save magnitude map
cv2.imshow('Mag Lenna',magL)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_2_edge_raw_lenna.png',magL)

# shapes.png
print('++++++mag shapes+++++')
filtered_S = cross_correlation_2d(shapes, filter)
magS, dirS = compute_image_gradient(filtered_S)
# show and save magnitude map
cv2.imshow('Mag Shapes',magS)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_2_edge_raw_shapes.png',magS)

def non_maximum_suppresion_dir(mag, dir):
    start = time.time()
    dir = np.rad2deg(dir) + 180
    m, n = mag.shape
    suppressed_mag = mag.copy()

    # gradient dir -> 8 bins
    for i in range(1, m-1):
      for j in range(1, n-1):

          center = mag[i][j]
          if (22.5< dir[i][j]<= 67.5) or (202.5< dir[i][j]<= 247.5):
                side = max(mag[i-1][j-1], mag[i+1][j+1])
          elif (67.5< dir[i][j] <= 112.5) or (247.5< dir[i][j] <= 292.5):
                side = max(mag[i-1][j], mag[i+1][j])
          elif (112.5< dir[i][j] <= 157.5) or (292.5< dir[i][j] <= 337.5):
                side = max(mag[i-1][j+1], mag[i+1][j-1])
          else:
                side = max(mag[i][j-1], mag[i][j+1])

          if center < side:
                suppressed_mag[i][j] = 0

    end = time.time() - start
    print('computational time of NMS:', end)

    return suppressed_mag

# lenna.png
print('++++++sup lenna+++++')
suppressed_L = non_maximum_suppresion_dir(magL, dirL)
# show and save suppresed magnitude map
cv2.imshow('Suppresed Lenna',suppressed_L)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_2_edge_sup_lenna.png',suppressed_L)

# shapes.png
print('++++++sup shapes+++++')
suppressed_S = non_maximum_suppresion_dir(magS, dirS)
# show and save suppresed magnitude map
cv2.imshow('Suppresed Shapes',suppressed_S)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./result/part_2_edge_sup_shapes.png',suppressed_S)
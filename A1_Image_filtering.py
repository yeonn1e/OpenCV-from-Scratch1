import numpy as np
import cv2
import time

# 1-1. Image Filetering by Cross-Correlation

lenna = cv2.imread('lenna.png' , cv2.IMREAD_GRAYSCALE)
shapes = cv2.imread('shapes.png' , cv2.IMREAD_GRAYSCALE)

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

def cross_correlation_1d(img, kernel):
  filtered_img = np.zeros((img.shape[0], img.shape[1]))
  kernelsize = kernel.shape[0]
  pad = kernelsize//2

  if kernel.ndim == 1: # 수평
    padded = np.zeros((img.shape[0]+2*pad,img.shape[1])) # 열 추가

    up = img[0, :].copy() #가장 위 행
    down = img[-1, :].copy() # 가장 아래 행
    padded[pad:pad+img.shape[0],:] = img

    for i in range(pad):
        padded[i, :] = up
        padded[-(i+1), :] = down
    kernel = kernel.reshape(1, kernel.shape[0])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            convolution = padded[i:i + kernelsize, j]
            filtering = np.sum(convolution * kernel)
            filtered_img[i, j] = filtering

  else: # 수직

    padded = np.zeros((img.shape[0],img.shape[1]+2*pad)) # 행 추가
    padded[:, pad:pad+img.shape[1]] = img
    left = img[:, 0].reshape((img.shape[0], )).copy() # 제일 왼쪽 열 -> 열벡터 형태 reshape
    right = img[:, -1].reshape((img.shape[0], )).copy() # 제일 오른쪽 열

    for i in range(pad):
        padded[:, i] = left
        padded[:, -(i+1)] = right

    kernel = kernel.reshape(1, kernel.shape[0])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            convolution = padded[i, j:j + kernelsize]
            filtering = np.sum(convolution * kernel)
            filtered_img[i, j] = filtering

  return filtered_img

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

# 1-2. The Gaussian FIlter

def gaussian(i,j,sigma):
  return (1/(2 * np.pi * sigma**2)) * np.exp(-1*((i**2 + j**2)/(2*(sigma**2))))

def get_gaussian_filter_1d(size, sigma):
  half = size // 2
  kernel = gaussian(np.arange(-half,half+1),0,sigma)
  kernel /= np.sum(kernel)

  return kernel

def get_gaussian_filter_2d(size, sigma):
  half = size // 2
  px, py = np.meshgrid(np.arange(-half,half+1), np.arange(-half,half+1))
  kernel = gaussian(px,py,sigma)
  kernel /= np.sum(kernel)

  return kernel

# 가우시안 적용 결과
kernel_1d = get_gaussian_filter_1d(5,1)
kernel_2d = get_gaussian_filter_2d(5,1)
print('kernel_1d:',kernel_1d, '\nkernel_2d:',kernel_2d)

"""# Show Image"""

def combine(img, kernels_lst, sigs_lst):
  x, y = img.shape[0], img.shape[1]
  output = np.zeros((img.shape[0]*3, img.shape[1]*3),dtype=np.float32)

  for i, kernel in enumerate(kernels_lst):
    for j, sig in enumerate(sigs_lst):

      filter = get_gaussian_filter_2d(kernel, sig)
      filtered = cross_correlation_2d(img, filter)
      output[i*x:x*(i+1), j*y:y*(j+1)] = filtered[:x,:y]

      cv2.putText(output, f'{kernel}x{kernel} s={sig}',(30+j*y ,30+i*x), cv2.FONT_ITALIC,
                  0.8, (0,0,0),2)

  return output

#lenna

kernels = [5,11,17]
sigs = [1,6,11]

# 9개의 이미지로 합침
lenna_imgs = combine(lenna, kernels, sigs)

cv2.imshow('Nine Lenna',lenna_imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 하나의 파일로 저장
cv2.imwrite('./result/part_1_gaussian_filtered_lenna.png',lenna_imgs)

#shapes

shapes_imgs = combine(shapes, kernels, sigs)
cv2.imshow('Nine shapes',shapes_imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 하나의 파일로 저장
cv2.imwrite('./result/part_1_gaussian_filtered_shapes.png',shapes_imgs)

# lenna
print('++++++lenna.png++++++')
# 사이즈 17, 시그마 11

# 1D kernels 수직, 수평으로 수행
filter_1d_hor = get_gaussian_filter_1d(17, 11)
filter_1d_ver = filter_1d_hor.reshape(17,1)

start_1d = time.time()

do_1d = cross_correlation_1d(lenna, filter_1d_hor)
first = cross_correlation_1d(do_1d, filter_1d_ver)
end_1d = time.time() - start_1d

# 2D kernel로 수행
filter_2d = get_gaussian_filter_2d(17,11)
start_2d = time.time()
do_2d = cross_correlation_2d(lenna, filter_2d)
end_2d = time.time() - start_1d

# 두 커널의 수행 시간 출력
print('size: 17, sigma: 11')
print('Computational times of 1D:',end_1d, ', Computational times of 2D:',end_2d)

# visualize a pixel-wise difference map
difference = do_2d - first
print('a pixel-wise difference map:\n')
cv2.imshow('difference map of lenna',difference)
cv2.waitKey(0)
cv2.destroyAllWindows()

# report the sum of (absolute) intensity differences to the console
print('sum of  intensity differences:',np.abs(difference).sum())

# shapes
print('++++++shapes.png++++++')
# 사이즈 17, 시그마 11

start_1d = time.time()
do_1d = cross_correlation_1d(shapes, filter_1d_hor)
first = cross_correlation_1d(do_1d, filter_1d_ver)
end_1d = time.time() - start_1d

# 2D kernel로 수행
filter_2d = get_gaussian_filter_2d(17, 11)
start_2d = time.time()
do_2d = cross_correlation_2d(shapes, filter_2d)
end_2d = time.time() - start_1d

# 두 커널의 수행 시간 출력
print('size: 17, sigma: 11')
print('Computational times of 1D:',end_1d, ', Computational times of 2D:',end_2d)

# visualize a pixel-wise difference map
difference = do_2d - first
print('a pixel-wise difference map:\n')
cv2.imshow('difference map of shapes',difference)
cv2.waitKey(0)
cv2.destroyAllWindows()


# report the sum of (absolute) intensity differences to the console
print('sum of  intensity differences:',np.abs(difference).sum())
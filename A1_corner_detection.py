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

def sobel(img):
    sobelX = np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
    sobelY = sobelX.T # transpose

    operate_x = cross_correlation_2d(img, sobelX)
    operate_y = cross_correlation_2d(img, sobelY)

    return operate_x, operate_y

def compute_corner_response(img):
    start = time.time()
    window = np.ones((5,5))
    K = 0.04

    # apply sobel filters
    dx, dy = sobel(img)
    # computing the second moment matrix
    covmat = np.array([[dx**2, dx*dy],
                   [dy*dx, dy**2]])

    dxdx = cross_correlation_2d(covmat[0][0],window)
    dydy = cross_correlation_2d(covmat[1][1],window)
    dxdy = cross_correlation_2d(covmat[0][1],window)

    # computing response values
    detM = dxdx*dydy -  dxdy**2
    trace = (dxdx+dxdy)
    R = detM - K*(trace**2)

    # update all negative values to 0 -> normalize to 0~1
    R = np.where(R < 0, 0, R)
    R = (R - np.min(R)) / (np.max(R)-np.min(R))

    end = time.time() - start

    print('computational time of computing corner response:', end)

    return R*255

filter = get_gaussian_filter_2d(7, 1.5)

# lenna.png
print('+++++corners of lenna++++++')
filtered_L = cross_correlation_2d(lenna, filter)
corner_L = compute_corner_response(filtered_L)
cv2.imshow('corners of lenna',corner_L)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_3_corner_raw_lenna.png',corner_L)

# shapes.png
print('+++++corners of shapes++++++')
filtered_S = cross_correlation_2d(shapes, filter)
corner_S = compute_corner_response(filtered_S)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('corners of shapes',corner_S)
cv2.imwrite('./result/part_3_corner_raw_shapes.png',corner_S)

def change_cornercolor(colored_img, response, threshold=0.1):
    colored_img[response>threshold] = [0,255,0] # green

    return colored_img

def non_maximum_suppression_win(R, winSize=11):
    start = time.time()
    threshold = 0.1
    response = R / 255

    suppressed_R = response.copy()
    suppressed_R[suppressed_R<threshold] = 0

    # check if max and centered
    for i in range(suppressed_R.shape[0] - winSize):
        for j in range(suppressed_R.shape[1] - winSize):
            max = np.max(suppressed_R[i:i + winSize, j:j + winSize])
            center = suppressed_R[i + winSize // 2, j + winSize // 2]
            if max != center:
                suppressed_R[i + winSize // 2, j + winSize // 2] = 0

    end = time.time() - start

    print('computational time of NMS window:', end)
    return suppressed_R

def circling(colored_img, nms_img):
    circled_img = colored_img.copy()
    for x, y in zip(*np.where(nms_img > 0)):
        circled_img = cv2.circle(circled_img, (y, x), 5, (0, 255, 0), 2)

    return circled_img

# lenna.png
threshold = 0.1
print('+++++green corners of lenna++++++')
colored_L = cv2.cvtColor(lenna, cv2.COLOR_GRAY2RGB)
greened_L = change_cornercolor(colored_L, corner_L/255)
cv2.imshow('green corners of lenna',greened_L)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_3_corner_bin_lenna.png',greened_L)

# shapes.png
print('+++++green corners of shapes++++++')
colored_S = cv2.cvtColor(shapes, cv2.COLOR_GRAY2RGB)
greened_S = change_cornercolor(colored_S, corner_S/255)
cv2.imshow('green corners of shapes',greened_S)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_3_corner_bin_shapes.png',greened_S)

print('+++++green circled lenna++++++')
winsup_L = non_maximum_suppression_win(corner_L,) # winsize 11 is default
colored_L = cv2.cvtColor(lenna, cv2.COLOR_GRAY2RGB)
circled_L = circling(colored_L,winsup_L)
cv2.imshow('green circled lenna',circled_L)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_3_corner_sup_lenna.png',circled_L)

print('+++++green circled shapes++++++')
winsup_S = non_maximum_suppression_win(corner_S,) # winsize 11 is default
colored_S = cv2.cvtColor(shapes, cv2.COLOR_GRAY2RGB)
circled_S = circling(colored_S, winsup_S)
cv2.imshow('green circled shapes',circled_S)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('./result/part_3_corner_sup_shapes.png',circled_S)


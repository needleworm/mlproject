import numpy as np
import scipy as sp
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import matplotlib.pyplot as plt

GRAY_MODE = 1
RGB_MODE = 3
ref_file = 'pokemon1024.png'

# Version
version = RGB_MODE

if version == GRAY_MODE :
    ref_img = sp.misc.imread(ref_file, flatten=True) # Read an image from a file as an array.
    # blurring for demo
    sigma = 3
    blurred_img = sp.ndimage.gaussian_filter(ref_img, sigma=sigma)
    sp.misc.imsave('blurred_img.png',blurred_img)
    blurred_img = sp.misc.imread('blurred_img.png', flatten=True)
    #save image
    sp.misc.imsave('blurred_img_gray.png',blurred_img)
    blurred_img = sp.misc.imread('blurred_img_gray.png', flatten=True)

    fig, axes  =plt.subplots(nrows=1, ncols=2, figsize=(10,4),
                             sharex=True, sharey=True,
                             subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()

    psnr_none = float("inf")
    ssim_none = ssim(ref_img, ref_img, data_range=ref_img.max() - ref_img.min())
    psnr_blur = psnr(ref_img,blurred_img,data_range=ref_img.max() - ref_img.min())
    ssim_blur = ssim(ref_img, blurred_img, data_range=blurred_img.max() - blurred_img.min())

    label = 'PSNR: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(ref_img)
    ax[0].set_xlabel(label.format(psnr_none, ssim_none))
    ax[0].set_title('Original image')

    ax[1].imshow(blurred_img)
    ax[1].set_xlabel(label.format(psnr_blur, ssim_blur))
    ax[1].set_title('Blurred Image')

    plt.tight_layout()
    plt.show()

elif version == RGB_MODE :
    # RGB version
    ref_img = sp.misc.imread(ref_file,mode='RGB') # Read an image from a file as an array.
    # blurring for demo
    r_pixels, g_pixels, b_pixels = np.dsplit(ref_img, 3)
    sigma = 3
    r_pixels = sp.ndimage.gaussian_filter(r_pixels, sigma=sigma)
    g_pixels = sp.ndimage.gaussian_filter(g_pixels, sigma=sigma)
    b_pixels = sp.ndimage.gaussian_filter(b_pixels, sigma=sigma)
    blurred_img = np.dstack((r_pixels, g_pixels, b_pixels))
    #save image
    sp.misc.imsave('blurred_img_color.png',blurred_img)
    blurred_img = sp.misc.imread('blurred_img_color.png')

    fig, axes  =plt.subplots(nrows=1, ncols=2, figsize=(10,4),
                             sharex=True, sharey=True,
                             subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()

    psnr_none = float("inf")
    ssim_none = ssim(ref_img, ref_img, data_range=ref_img.max() - ref_img.min(),multichannel=True)
    psnr_blur = psnr(ref_img,blurred_img,data_range=ref_img.max() - ref_img.min())
    ssim_blur = ssim(ref_img, blurred_img, data_range=blurred_img.max() - blurred_img.min(),multichannel=True)

    label = 'PSNR: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(ref_img)
    ax[0].set_xlabel(label.format(psnr_none, ssim_none))
    ax[0].set_title('Original image')

    ax[1].imshow(blurred_img)
    ax[1].set_xlabel(label.format(psnr_blur, ssim_blur))
    ax[1].set_title('Blurred Image')

    plt.tight_layout()
    plt.show()


from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import numpy as np


def psnr(batch_size, ref_image_set, output_image_set):
    psnr_val = 0
    ref_image_set = np.array(ref_image_set)
    output_image_set = np.array(output_image_set)

    if batch_size == 1:
        return compare_psnr(ref_image_set, output_image_set, data_range=ref_image_set.max() - ref_image_set.min())

    for i in range(batch_size):
        ref_image = ref_image_set[i]
        output_image = output_image_set[i]
        psnr_val += compare_psnr(ref_image, output_image, data_range=ref_image.max() - ref_image.min())

    psnr_val =  np.mean(psnr_val)
    return psnr_val


def ssim(batch_size, ref_image_set, output_image_set):
    ssim_val = 0
    ref_image_set = np.array(ref_image_set)
    output_image_set = np.array(output_image_set)

    if batch_size == 1:
        return compare_ssim(ref_image_set, output_image_set, data_range=ref_image_set.max() - ref_image_set.min(),
                            multichannel=True)

    for i in range(batch_size):
        ref_image = ref_image_set[i]
        output_image = output_image_set[i]
        ssim_val = compare_ssim(ref_image, output_image, data_range=ref_image.max() - ref_image.min(),
                                multichannel=True)

    ssim_val = np.mean(ssim_val)

    return ssim_val




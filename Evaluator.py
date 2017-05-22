
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


def ssim(ref_image, output_image):
    ssim_val = compare_ssim(ref_image, output_image, data_range=ref_image.max() - ref_image.min(),multichannel=True)
    return ssim_val


def psnr(ref_image, output_image):
    psnr_val = compare_psnr(ref_image, output_image, data_range=ref_image.max() - ref_image.min())
    return psnr_val


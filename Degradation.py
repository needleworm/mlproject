from skimage import exposure, filters, io, img_as_float, transform, util
import numpy as np
import matplotlib.pyplot as plt


def degrade(image, degrade_type):
    degrade_image = np.zeros(image.shape)
    degrade_image[:]= im2double(image)

    if ("blur" in degrade_type):
        # K = fspecial('disk', 7) ; circular averaging filter
        # IBlur = imfilter(I, K, 'conv', 'replicate');
        # IBlur = uint8((IBlur)*255+0.5);
        # IBlur = imnoise(IBlur, 'gaussian', 0, 0.0005);
        degrade_image = filters.gaussian(degrade_image, sigma=7, multichannel=True)
        degrade_image = util.random_noise(degrade_image, mode='gaussian', clip=True, mean=0, var=0.0005)

    if ("saturate" in degrade_type):
        # I = im2double(I) * 1.3;
        degrade_image = img_as_float(degrade_image) * 1.3
        degrade_image = exposure.rescale_intensity(degrade_image, in_range=(0, 1))

    if("downscale" in degrade_type):
        # downscale by 4
        scale = 4
        H, W, C = degrade_image.shape
        H = H - np.mod(H, scale)
        W = W - np.mod(W, scale)
        degrade_image = degrade_image[0:H, 0:W, :]
        degrade_image = transform.rescale(degrade_image, (1./scale))
        degrade_image = transform.rescale(degrade_image, (scale/1.))

    if ("compress" in degrade_type):
        # % jpeg compression
        # imwrite(IBlur, 'blur.jpg', 'Quality', 70);
        io.imsave("images/compress.jpeg", degrade_image, plugin='pil', quality=60)
        degrade_image = io.imread("images/compress.jpeg")
        degrade_image = im2double(degrade_image)

    return degrade_image

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
'''
def main():
    fig, axes = plt.subplots(nrows=1, ncols=5,
                             sharex=True, sharey=True)
    ax = axes.ravel()

    image = io.imread("images/genzi.jpg")
    ax[0].imshow(im2double(image))
    blur_image = degrade(image, ["blur"])
    ax[1].imshow(blur_image)
    saturate_image = degrade(image, ["saturate"])
    ax[2].imshow(saturate_image)
    downscale_image = degrade(image, ["downscale"])
    ax[3].imshow(downscale_image)
    compress_image = degrade(image, ["compress"])
    ax[4].imshow(compress_image)
    plt.tight_layout()
    plt.show()


main()
'''

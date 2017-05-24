from PIL import Image, ImageFilter
import os
import scipy
import scipy.misc
import skimage
import numpy as np

def degrade(image, degrade_type):
    if ("blur" in degrade_type):
        # K = fspecial('disk', 7) ; circular averaging filter
        # IBlur = imfilter(I, K, 'conv', 'replicate');
        # IBlur = uint8((IBlur)*255+0.5);
        # IBlur = imnoise(IBlur, 'gaussian', 0, 0.0005);

        image = (scipy.ndimage.filters.gaussian_filter(image, sigma=7)*255+0.5).astype("uint8")
        image = image + np.random.gaussian(0, 0.0005, image.shape).astype(float)

    if ("saturation" in degrade_type):
        # I = im2double(I) * 1.3;
        image = skimage.img_as_float(image)*1.3

    if("downgrade" in degrade_type):
        image = skimage.transform.rescale(image, 0.25)
        image = skimage.transform.rescale(image, 4)

    if ("compression" in degrade_type):
        # % jpeg compression
        # imwrite(IBlur, 'blur.jpg', 'Quality', 70);
        skimage.io.imsave("images/compress.jpeg", image, quality=70)
        image = skimage.io.imread("images/compress.jpeg")

    return image


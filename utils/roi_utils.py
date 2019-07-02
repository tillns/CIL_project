import os
import csv
import random
import numpy as np
import numpy.ma as ma
import PIL.Image
import sys
import cv2
import math
import yaml
import pywt
try:
    import compute_hist
    use_grad = True
except ModuleNotFoundError:
    use_grad = False


def roi_histograms(image, conf):
    roi_conf = conf['ROI_options']
    np_image = np.array(image, dtype=np.uint8)
    range_normal = (0, 255)
    eps = np.power(10.0, -15.0)
    psd_fft = np.fft.fft2(np.asarray(image, dtype=np.float32))
    psd_fft_shift = np.abs(np.fft.fftshift(psd_fft))**2
    psd_log = 10 * np.log10(psd_fft_shift + eps)
    psd_noshifted = np.abs(psd_fft)**2
    psd_noshifted_log = 10 * np.log10(psd_noshifted + eps)
    ranges_fft = [(range_fft[0], range_fft[1]) for range_fft in conf['ranges']]

    hists = []

    if roi_conf['whole_img']['include']:
        if not roi_conf['whole_img']['prepr_fft']:
            hists.append(np.histogram(np_image, bins=roi_conf['whole_img']['num_bins'], range=range_normal)[0])
        else:
            for num_range, range_fft in enumerate(ranges_fft):
                hists.append(np.histogram(psd_log, bins=roi_conf['whole_img']['num_bins'][num_range], range=range_fft)[0])


    # example for subset of frequencies (the frequency spectrum is very symmetric)
    # adding these additional histograms gave an improvement of about 5 %
    if roi_conf['quarter_img']['include']:
        num_rows, num_cols = psd_fft_shift.shape

        for num_range, range_fft in enumerate(ranges_fft):
            if roi_conf['quarter_img']['prepr_fft']:
                quarter_psd = psd_fft_shift[num_rows // 2 : num_rows, num_cols // 2 : num_cols]
                quarter = 10 * np.log10(quarter_psd + eps)
                quarter_range = range_fft

            else:
                quarter = np_image[num_rows // 2 : num_rows, num_cols // 2 : num_cols]
                quarter_range = range_normal

            image_features_4, _ = np.histogram(quarter[0:250, 250:500][0:125, 0:125],
                                               bins=roi_conf['quarter_img']['num_bins'][num_range],
                                               range=quarter_range)
            image_features_5, _ = np.histogram(quarter[0:250, 250:500][0:125, 125:250],
                                               bins=roi_conf['quarter_img']['num_bins'][num_range],
                                               range=quarter_range)
            image_features_6, _ = np.histogram(quarter[0:250, 250:500][125:250, 0:125],
                                               bins=roi_conf['quarter_img']['num_bins'][num_range],
                                               range=quarter_range)
            image_features_7, _ = np.histogram(quarter[0:250, 250:500][125:250, 125:250],
                                               bins=roi_conf['quarter_img']['num_bins'][num_range],
                                               range=quarter_range)
            hists.extend([image_features_4, image_features_5, image_features_6, image_features_7])

    if roi_conf['radial']['include']:
        for num_rad, radius in enumerate(roi_conf['radial']['radii']):
            mask = _create_circular_mask(1000, 1000, [500, 500], radius)
            for num_range, range_fft in enumerate(ranges_fft):
                if roi_conf['radial']['prepr_fft']:
                    if roi_conf['radial']['shift_fft']:
                        hists.append(_compute_histogram_from_mask(mask, psd_log,
                                                                  roi_conf['radial']['num_bins'][num_rad][num_range],
                                                                  range_fft))
                    else:
                        hists.append(_compute_histogram_from_mask(mask, psd_noshifted_log,
                                                                  roi_conf['radial']['num_bins'][num_rad][num_range],
                                                                  range_fft))
                else:
                    hists.append(_compute_histogram_from_mask(mask, np_image,
                                                              roi_conf['radial']['num_bins'][num_rad][num_range],
                                                              range_normal))

    if roi_conf['grad']['include'] and use_grad:
        # Todo: I think this only uses data without fft, add fft

        grad_img = ((psd_log + 150.0) / 2.0).astype(np.uint8)
        gx = cv2.Sobel(grad_img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(grad_img, cv2.CV_32F, 0, 1, ksize=1)

        msk = _create_circular_mask(1000, 1000, [500, 500], radius=roi_conf['grad']['radius'])

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        mag_mskd = ma.masked_array(mag, mask=msk)
        angle_mskd = ma.masked_array(angle, mask=msk)

        grad_hist = np.zeros(roi_conf['grad']['num_bins']).astype(np.float32)

        compute_hist.compute_hist_func(mag_mskd[0:500,500:1000].compressed().flatten(),
                                       angle_mskd[0:500,500:1000].compressed().flatten(),
                                       grad_hist, angle_mskd[0:500,500:1000].compressed().size, roi_conf['grad']['num_bins'])

        hists.append(grad_hist)

    if roi_conf['angle']['include']:
        for num_it, num_bins in enumerate(roi_conf['angle']['num_bins']):
            for num_range, range_fft in enumerate(ranges_fft):
                mask = _create_angle_mask(1000, 1000, roi_conf['angle']['angle_l'][num_it],
                                          roi_conf['angle']['angle_h'][num_it])
                hist = _compute_histogram_from_mask(mask, psd_log, num_bins[num_range], range_fft) \
                    if roi_conf['angle']['prepr_fft'] else \
                    _compute_histogram_from_mask(mask, np_image, num_bins, range_normal)
                hists.append(hist)
    return hists


def _compute_histogram_from_mask(mask, image, num_bins, range):
    """Computes a histogram for the image region defined by the mask for each channel

    Parameters
    ----------
    mask : numpy array
        boolean mask. Shape (H, W).
    image : numpy array
        original image. Shape (H, W, C).
    num_bins : int
        the bins argument for the histogram
    range : tuple
        the range argument for the histogram


    Returns
    -------
    hist : list
        list of length bins, containing the histogram of the masked image values
    """

    # Apply binary mask to your array, you will get array with shape (N, C)
    region = image[mask]

    hist, _ = np.histogram(region, bins=num_bins, range=range)
    return hist


def _create_circular_mask(h, w, center=None, radius=None, invert_mask=False):
    """ Computes a circular mask

    Get a circular mask on an image given its height, width and optionally center and radius of the
    mask.

    Paramters
    ---------
    h : int
        height of the image
    w : int
        width of the image
    center : tuple
        tuple of ints, representing the center of the generated mask
    radius : int or tuple
        either the radius of a circular mask or the min- and max-radius of the mask
    invert_mask : bool
        wheter the generated mask should be inverted before returning it

    Returns
    -------
    mask : numpy array
        a numpy array of boolean values
    """

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if isinstance(radius, list) and len(radius) == 2:
        if radius[0] < 0:
            radius[0] = np.abs(radius[0])
            radius[1] = np.abs(radius[1])
            invert_mask = True
        mask = (max(radius) <= dist_from_center) | (dist_from_center <= min(radius)) if invert_mask else \
            (max(radius) >= dist_from_center) & (dist_from_center >= min(radius))
    else:
        if radius < 0:
            radius = np.abs(radius)
            invert_mask = True
        mask = dist_from_center >= radius if invert_mask else dist_from_center <= radius
    return mask


def _create_angle_mask(h, w, angle_l, angle_h):

    """ Computes a mask for an interval of angles

    Returns an image mask masking all points which are not at an angle inside [angle_l, angle_h] or
    [angle_l - 180.0, angle_h - 180.0] to the center of the image. The angles are measured counter-clockwise
    from the horizontal.


    Paramters
    ---------
    h : int
        The image height
    w : int
        The image width
    angle_l : float
        The first angle of the interval. Has to be inside [0.0, angle_h].
    angle_h : float
        The second angle of the interval. Has to be inside [angle_l, 180.0].


    Returns
    -------
    mask : array-like
        The image mask

    """

    center = [w // 2, h // 2]

    Y, X = np.ogrid[:h, :w]


    angle = np.degrees(np.arctan2((-(Y - center[1])), (X-center[0])))


    mask_angle = ((angle_l <= angle) * (angle <=  angle_h)) + \
                    ((angle_l - 180.0 <= angle) * (angle <= angle_h - 180.0))

    mask_angle[w // 2 , h // 2] = True # center


    mask_angle = np.invert(mask_angle)

    return mask_angle

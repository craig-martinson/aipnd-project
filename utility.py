import numpy as np


def print_elapsed_time(total_time):
    ''' Prints elapsed time in hh:mm:ss format
    '''
    hh = int(total_time / 3600)
    mm = int((total_time % 3600) / 60)
    ss = int((total_time % 3600) % 60)
    print(
        "\n** Total Elapsed Runtime: {:0>2}:{:0>2}:{:0>2}".format(hh, mm, ss))


def resize_image(image, size):
    ''' Resize image so shortest side is 'size' pixels,
        maintain aspect ratio  
    '''
    w, h = image.size

    if h > w:
        # Set width to 'size' and scale height to maintain aspect ratio
        h = int(max(h * size / w, 1))
        w = int(size)
    else:
        # Set height to 'size' and scale width to maintain aspect ratio
        w = int(max(w * size / h, 1))
        h = int(size)

    return image.resize((w, h))


def crop_image(image, size):
    ''' Return a cropped square region from centre of image
    '''
    w, h = image.size
    x0 = (w - size) / 2
    y0 = (h - size) / 2
    x1 = x0 + size
    y1 = y0 + size

    return image.crop((x0, y0, x1, y1))


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Resize image so shortest side is 256 pixels
    resized_image = resize_image(image, 256)

    # Crop image
    cropped_image = crop_image(resized_image, 224)

    # Convert image to float array
    np_image = np.array(cropped_image) / 255.

    # Normalize array
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # PyTorch tensors assume the color channel is the first dimension
    # but PIL assumes is the third dimension
    np_image = np_image.transpose((2, 0, 1))

    return np_image

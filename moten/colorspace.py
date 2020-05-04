'''Code to convert RGB to other color spaces
'''
#
# Color transformations from
# /auto/k1/shinji/matlab/colorspace/colorspace2.m
# /auto/k1/shinji/matlab/strflab_adds/preprocColorSpace.m
#
# Code originally by Pascal Getreuer (https://getreuer.info/) 2005-2006:
# https://www.mathworks.com/matlabcentral/fileexchange/28790-colorspace-transformations
#
# Only a few transformations were converted to python.
#
# Anwar O. Nunez-Elizalde (2016).
#
# Updates:
#
#
import numpy as np


def rgb2lab(image):
    '''Convert RGB to CIE LAB color space **in-place**

    Parameters
    ----------
    image : 3D numpy float array
        Array must be in [0,1] range. Last
        dimension corresponds to RGB channels.

    Returns
    -------
    LAB : 3D numpy float array
        The CIE LAB representation in the image.
    '''

    WhitePoint = np.asarray([0.95047,1.0,1.08883])
    # Convert image to XYZ
    image = rgb2xyz(image)
    # % Convert XYZ to CIE L*a*b*
    X = image[:,:,0]/WhitePoint[0]
    Y = image[:,:,1]/WhitePoint[1]
    Z = image[:,:,2]/WhitePoint[2]
    fX = _ff(X)
    fY = _ff(Y)
    fZ = _ff(Z)
    image[:,:,0] = 116.0*fY - 16          # L*
    image[:,:,1] = 500.0*(fX - fY)        # a*
    image[:,:,2] = 200.0*(fY - fZ)        # b*
    return image


def rgb2lch(image):
    '''Convert RGB to CIE LCH color space **in-place**

    Parameters
    ----------
    image : 3D numpy float array
        Array must be in [0,1] range. Last
        dimension corresponds to RGB channels.

    Returns
    -------
    LCH : 3D numpy float array
        The CIE LCH representation in the image.
    '''

    image = rgb2lab(image)
    H = np.arctan2(image[:,:,2],image[:,:,1])
    H = H*180/np.pi + 360*(H < 0.0)
    image[:,:,1] = np.sqrt(image[:,:,1]**2 + image[:,:,2]**2)  # C
    image[:,:,2] = H                                           # H
    return image


def rgb2xyz(image):
    '''Convert RGB to CIE XYZ color space **in-place**

    Parameters
    ----------
    image : 3D numpy float array
        Array must be in [0,1] range. Last
        dimension corresponds to RGB channels.

    Returns
    -------
    LAB : 3D numpy float array
        The CIE XYZ representation in the image.
    '''
    # % Undo gamma correction
    R = _invgammacorrection(image[:,:,0])
    G = _invgammacorrection(image[:,:,1])
    B = _invgammacorrection(image[:,:,2])
    # % Convert RGB to XYZ
    xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                            [0.212671, 0.715160, 0.072169],
                            [0.019334, 0.119193, 0.950227]])
    T = xyz_from_rgb.T.ravel()
    image[:,:,0] = T[0]*R + T[3]*G + T[6]*B # X
    image[:,:,1] = T[1]*R + T[4]*G + T[7]*B # Y
    image[:,:,2] = T[2]*R + T[5]*G + T[8]*B # Z
    return image


def _ff(Y):
    '''Obscure helper function
    '''
    fY = np.real(Y**(1./3.))
    idx = Y < 0.008856
    fY[idx] = Y[idx]*(7.787) + (4./29.)
    return fY


def _invgammacorrection(Rp):
    '''
    '''
    R = np.real(((Rp + 0.055)/1.055)**(2.4))
    idx = Rp < 0.04045
    R[idx] = Rp[idx] / 12.92
    return R


def _gamma_correct(image, gamma=1.0):
    """Do not use unless you know what you're doing
    """
    if gamma != 1.0:
        if not (0 <= image.min() and 1 >= image.max()):
            image /= 255.
        image = image**gamma
    return image


if __name__ == '__main__':
    pass

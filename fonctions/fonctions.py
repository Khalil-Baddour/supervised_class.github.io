# @author: marc lang, modifié par Khalil BADDOUR

from osgeo import gdal
import numpy as np
import geopandas as gpd

def open_image(filename, verbose=False):
  """
  Ouvrir le fichier image avec  gdal

  osgeo.gdal.Dataset
  """
  data_set = gdal.Open(filename, gdal.GA_ReadOnly)

  if data_set is None:
      print("Impossible d'ouvrir {}".format(filename))
      exit()
  elif data_set is not None and verbose:
      print('{} a été ouverte'.format(filename))

  return data_set



def get_image_dimension(data_set, verbose=False):
    """
    Pour obtenir les dimensions de l'image'

    Parameters
    ----------
    data_set : osgeo.gdal.Dataset

    Returns
    -------
    nb_col : int
    nb_lignes : int
    nb_band : int
    """

    nb_col = data_set.RasterXSize
    nb_lignes = data_set.RasterYSize
    nb_band = data_set.RasterCount
    if verbose:
        print('Nombre de colonnes :', nb_col)
        print('Nombre de lignes :', nb_lignes)
        print('Nombre de bandes :', nb_band)

    return nb_col, nb_lignes, nb_band


def convert_data_type_from_gdal_to_numpy(gdal_data_type):
    """
    pour convertir le type de données de gdal en numpy

    Parameters
    ----------
    gdal_data_type : str
        Data type with gdal syntax
    Returns
    -------
    numpy_data_type : str
        Data type with numpy syntax
    """
    if gdal_data_type == 'Byte':
        numpy_data_type = 'uint8'
    else:
        numpy_data_type = gdal_data_type.lower()
    return numpy_data_type


def get_samples_from_roi(raster_name, roi_name, value_to_extract=None,
                         bands=None, output_fmt='full_matrix'):
    '''
    The function get the set of pixel of an image according to an roi file
    (raster). In case of raster format, both map should be of same
    size.

    Parameters
    ----------
    raster_name : string
        The name of the raster file, could be any file GDAL can open
    roi_name : string
        The path of the roi image.
    value_to_extract : float, optional, defaults to None
        If specified, the pixels extracted will be only those which are equal
        this value. By, defaults all the pixels different from zero are
        extracted.
    bands : list of integer, optional, defaults to None
        The bands of the raster_name file whose value should be extracted.
        Indexation starts at 0. By defaults, all the bands will be extracted.
    output_fmt : {`full_matrix`, `by_label` }, (optional)
        By default, the function returns a matrix with all pixels present in the
        ``roi_name`` dataset. With option `by_label`, a dictionnary
        containing as many array as labels present in the ``roi_name`` data
        set, i.e. the pixels are grouped in matrices corresponding to one label,
        the keys of the dictionnary corresponding to the labels. The coordinates
        ``t`` will also be in dictionnary format.

    Returns
    -------
    X : ndarray or dict of ndarra
        The sample matrix. A nXd matrix, where n is the number of referenced
        pixels and d is the number of variables. Each line of the matrix is a
        pixel.
    Y : ndarray
        the label of the pixel
    t : tuple or dict of tuple
        tuple of the coordinates in the original image of the pixels
        extracted. Allow to rebuild the image from `X` or `Y`
    '''

    # Get size of output array
    raster = open_image(raster_name)
    nb_col, nb_row, nb_band = get_image_dimension(raster)

    # Get data type
    band = raster.GetRasterBand(1)
    gdal_data_type = gdal.GetDataTypeName(band.DataType)
    numpy_data_type = convert_data_type_from_gdal_to_numpy(gdal_data_type)

    # Check if is roi is raster or vector dataset
    roi = open_image(roi_name)

    if (raster.RasterXSize != roi.RasterXSize) or \
            (raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        print('Raster : {}'.format(raster_name))
        print('Roi : {}'.format(roi_name))
        exit()

    if not bands:
        bands = list(range(nb_band))
    else:
        nb_band = len(bands)

    #  Initialize the output
    ROI = roi.GetRasterBand(1).ReadAsArray()
    if value_to_extract:
        t = np.where(ROI == value_to_extract)
    else:
        t = np.nonzero(ROI)  # coord of where the samples are different than 0

    Y = ROI[t].reshape((t[0].shape[0], 1)).astype('int32')

    del ROI
    roi = None  # Close the roi file

    try:
        X = np.empty((t[0].shape[0], nb_band), dtype=numpy_data_type)
    except MemoryError:
        print('Impossible to allocate memory: roi too large')
        exit()

    # Load the data
    for i in bands:
        temp = raster.GetRasterBand(i + 1).ReadAsArray()
        X[:, i] = temp[t]
    del temp
    raster = None  # Close the raster file

    # Store data in a dictionnaries if indicated
    if output_fmt == 'by_label':
        labels = np.unique(Y)
        dict_X = {}
        dict_t = {}
        for lab in labels:
            coord = np.where(Y == lab)[0]
            dict_X[lab] = X[coord]
            dict_t[lab] = (t[0][coord], t[1][coord])

        return dict_X, Y, dict_t
    else:
        return X, Y, t,
    
    
    
    
def write_image(out_filename, array, data_set=None, gdal_dtype=None,
                transform=None, projection=None, driver_name=None,
                nb_col=None, nb_ligne=None, nb_band=None):
    """
    écrire un tableau dans une image

    Parameters
    ----------
    out_filename : str
        Path of the output image.
    array : numpy.ndarray
        Array to write
    nb_col : int (optional)
        If not indicated, the function consider the `array` number of columns
    nb_ligne : int (optional)
        If not indicated, the function consider the `array` number of rows
    nb_band : int (optional)
        If not indicated, the function consider the `array` number of bands
    data_set : osgeo.gdal.Dataset
        `gdal_dtype`, `transform`, `projection` and `driver_name` values
        are infered from `data_set` in case there are not indicated.
    gdal_dtype : int (optional)
        Gdal data type (e.g. : gdal.GDT_Int32).
    transform : tuple (optional)
        GDAL Geotransform information same as return by
        data_set.GetGeoTransform().
    projection : str (optional)
        GDAL projetction information same as return by
        data_set.GetProjection().
    driver_name : str (optional)
        Any driver supported by GDAL. Ignored if `data_set` is indicated.
    Returns
    -------
    None
    """
    # Get information from array if the parameter is missing
    nb_col = nb_col if nb_col is not None else array.shape[1]
    nb_ligne = nb_ligne if nb_ligne is not None else array.shape[0]
    array = np.atleast_3d(array)
    nb_band = nb_band if nb_band is not None else array.shape[2]


    # Get information from data_set if provided
    transform = transform if transform is not None else data_set.GetGeoTransform()
    projection = projection if projection is not None else data_set.GetProjection()
    gdal_dtype = gdal_dtype if gdal_dtype is not None \
        else data_set.GetRasterBand(1).DataType
    driver_name = driver_name if driver_name is not None \
        else data_set.GetDriver().ShortName

    # Create DataSet
    driver = gdal.GetDriverByName(driver_name)
    output_data_set = driver.Create(out_filename, nb_col, nb_ligne, nb_band,
                                    gdal_dtype)
    output_data_set.SetGeoTransform(transform)
    output_data_set.SetProjection(projection)

    # Fill it and write image
    for idx_band in range(nb_band):
        output_band = output_data_set.GetRasterBand(idx_band + 1)
        output_band.WriteArray(array[:, :, idx_band])  # not working with a 2d array.
                                                       # this is what np.atleast_3d(array)
                                                       # was for
        output_band.FlushCache()

    del output_band
    output_data_set = None
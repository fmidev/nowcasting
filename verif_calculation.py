"""Methods for spatial and temporal interpolation of precipitation fields."""
# This script calculates verification statistics for fields
# VERIF METRICS:
# RMSE




try:
  import cv2
except ImportError:
  raise Exception("OpenCV Python bindings not found")
import numpy as np
from scipy.ndimage import gaussian_filter

def RMSE(obsfields, modelfields, mask_nodata):
  """This function calculates simple RMSE verif metric for forecast fields. The incoming data is assumed to have a form [time, x, y]. In other words, several time steps are allowed in the data.
  
  Parameters
  ----------
  obsfields : array-like
    Three-dimensional array (time, x, y) containing the observational fields.
  modelfields : array-like
    Three-dimensional array (time, x, y) containing the model fields.
  mask_nodata : array-like
    Three-dimensional array containing the nodata mask
  
  Returns
  -------
  out : array
    List of two-dimensional arrays containing the interpolated precipitation 
    fields ordered by time.
  """

  if obsfields.shape != modelfields.shape:
    raise ValueError("obsfields and modelfields have different shapes")

  result_array = np.ones(obsfields.shape[0])
  
  for time_step in np.arange(0,obsfields.shape[0]):
     obs = obsfields[time_step,~mask_nodata]
     model = modelfields[time_step,~mask_nodata]
     result = np.sqrt(((model - obs)**2).mean())
     result_array[time_step] = result
  
  return result_array






def ME(obsfields, modelfields, mask_nodata):
  """This function calculates simple ME verif metric for forecast fields. The incoming data is assumed to have a form [time, x, y]. In other words, several time steps are allowed in the data.
  
  Parameters
  ----------
  obsfields : array-like
    Three-dimensional array (time, x, y) containing the observational fields.
  modelfields : array-like
    Three-dimensional array (time, x, y) containing the model fields.
  mask_nodata : array-like
    Three-dimensional array containing the nodata mask
  
  Returns
  -------
  out : array
    List of two-dimensional arrays containing the interpolated precipitation 
    fields ordered by time.
  """

  if obsfields.shape != modelfields.shape:
    raise ValueError("obsfields and modelfields have different shapes")

  result_array = np.ones(obsfields.shape[0])
  
  for time_step in np.arange(0,obsfields.shape[0]):
     obs = obsfields[time_step,~mask_nodata]
     model = modelfields[time_step,~mask_nodata]
     result = (model - obs).mean()
     result_array[time_step] = result
  
  return result_array

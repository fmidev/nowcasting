"""Methods for spatial and temporal interpolation of precipitation fields."""
# This is slightly modified from interpolation.py: input fields can have several timesteps to them, also verification metrics are being calculated for the interpolated fields.

try:
  import cv2
except ImportError:
  raise Exception("OpenCV Python bindings not found")
from numpy import arange, dstack, float32, isfinite, log, logical_and, meshgrid, \
  nan, ones, ubyte
from scipy.ndimage import gaussian_filter

def advection(obsfields, modelfields, mask_nodata, farneback_params, R_min=0.1, R_max=30.0, 
              missingval=nan, logtrans=False, predictability, seconds_between_steps):
  """Temporal interpolation between two precipitation fields by using advection 
  field. The motion is estimated by using the Farneback algorithm implemented 
  in OpenCV.
  
  Parameters
  ----------
  obsfields : array-like
    Three-dimensional array (time, x, y) containing the observational fields.
  modelfields : array-like
    Three-dimensional array (time, x, y) containing the model fields.
  NOT USED n_interp_frames : int
    Number of frames to interpolate between the given precipitation fields.
  mask_nodata : array-like
    Three-dimensional array containing the nodata mask
  farneback_params : tuple
    Parameters for the Farneback optical flow algorithm, see the documentation 
    of the Python OpenCV interface.
  R_min : float
    Minimum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_min is the min value of all values contained in R1 and R2.
  R_max : float
    Maximum value for optical flow computations. For prec the thresholds are defined manually, for other variables R_max is the max value of all values contained in R1 and R2.
  missingval : float
    Value that is used for missing data. No interpolation is done for missing 
    values.
  logtrans : bool
    If True, logarithm is taken from R1 and R2 when computing the motion 
    vectors. This might improve the reliability of motion estimation.
  predictability : int
    Predictability in hours
  seconds_between_steps: int
    How long should two timesteps differ to each other?
  
  Returns
  -------
  out : array
    List of two-dimensional arrays containing the interpolated precipitation 
    fields ordered by time.
  """

  R1 = obsfields[0,:,:]
  R2 = modelfields[predictability,:,:]
  n_interp_frames = predictability*3600/seconds_between_steps - 1 # predictability is defined in hours
  mask_nodata_t = mask_nodata[0,:,:] # HERE ASSUMING THAT THE NODATA FIELDS ARE SIMILAR FOR ALL THE TIME STEPS!
  

  if R1.shape != R2.shape:
    raise ValueError("R1 and R2 have different shapes")
  
  X,Y = meshgrid(arange(R1.shape[1]), arange(R1.shape[0]))
  W = dstack([X, Y]).astype(float32)
  
  # Here changing to float to ubyte type. ubyte is an 8-bit unsigned integral data type (http://x10.sourceforge.net/x10doc/2.3.0/x10/lang/UByte.html), the 8-bit conversion of the data is necessary for the function cv2.calcOpticalFlowFarneback. Also smearing out image as defined by https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
  R1_f = _filtered_ubyte_image(R1, mask_nodata_t, R_min, R_max, filter_stddev=1.0, logtrans=logtrans)
  R2_f = _filtered_ubyte_image(R2, mask_nodata_t, R_min, R_max, filter_stddev=1.0, logtrans=logtrans)
  
  if int(cv2.__version__.split('.')[0]) == 2:
    VF = cv2.calcOpticalFlowFarneback(R1_f, R2_f, *farneback_params)
    VB = cv2.calcOpticalFlowFarneback(R2_f, R1_f, *farneback_params)
  else:
    VF = cv2.calcOpticalFlowFarneback(R1_f, R2_f, None, *farneback_params)
    VB = cv2.calcOpticalFlowFarneback(R2_f, R1_f, None, *farneback_params)
  
  tws = 1.0*arange(1, n_interp_frames + 1) / (n_interp_frames + 1)
  
  R_interp = []
  for tw in tws:
    R1_warped = cv2.remap(R1, W-tw*VF,       None, cv2.INTER_LINEAR)
    R2_warped = cv2.remap(R2, W-(1.0-tw)*VB, None, cv2.INTER_LINEAR)
    
    MASK1  = logical_and(isfinite(R1_warped), R1_warped != missingval)
    MASK2  = logical_and(isfinite(R2_warped), R2_warped != missingval)
    MASK12 = logical_and(MASK1, MASK2)
    
    R_interp_cur = ones(R1_warped.shape) * missingval
    R_interp_cur[MASK12] = (1.0-tw)*R1_warped[MASK12] + tw*R2_warped[MASK12]
    MASK_ = logical_and(MASK1, ~MASK2)
    R_interp_cur[MASK_] = R1_warped[MASK_]
    MASK_ = logical_and(MASK2, ~MASK1)
    R_interp_cur[MASK_] = R2_warped[MASK_]
    
    R_interp.append(R_interp_cur)
  
  return R_interp

# This function smooths out the image before calculating the motion vectors
def _filtered_ubyte_image(I, mask_nodata_t, R_min, R_max, filter_stddev=1.0, logtrans=False):
  I = I.copy()
  
  MASK = ~np.ma.getmask(mask_nodata_t)
  I[I > R_max] = R_max
  
  if logtrans == True:
    I = log(I)
    R_min = log(R_min)
    R_max = log(R_max)
  # Scale actual data points to have float values between 128...255. The missing values are assigned with the value of 0.
  I[MASK]  = 128.0 + (I[MASK] - R_min) / (R_max - R_min) * 127.0
  I[~MASK] = 0.0
  
  I = I.astype(ubyte)
  I = gaussian_filter(I, sigma=filter_stddev)

  return I

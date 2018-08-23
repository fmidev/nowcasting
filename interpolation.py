"""Methods for spatial and temporal interpolation of precipitation fields."""

try:
  import cv2
except ImportError:
  raise Exception("OpenCV Python bindings not found")
from numpy import arange, dstack, float32, isfinite, log, logical_and, meshgrid, \
  nan, ones, ubyte
from scipy.ndimage import gaussian_filter

def advection(R1, R2, n_interp_frames, farneback_params, R_min=0.1, R_max=30.0, 
              missingval=nan, logtrans=False):
  """Temporal interpolation between two precipitation fields by using advection 
  field. The motion is estimated by using the Farneback algorithm implemented 
  in OpenCV.
  
  Parameters
  ----------
  R1 : array-like
    Two-dimensional array containing the first precipitation field.
  R2 : array-like
    Two-dimensional array containing the second precipitation field.
  n_interp_frames : int
    Number of frames to interpolate between the given precipitation fields.
  farneback_params : tuple
    Parameters for the Farneback optical flow algorithm, see the documentation 
    of the Python OpenCV interface.
  R_min : float
    Minimum precipitation intensity for optical flow computations. Values below 
    R_min are set to zero.
  R_max : float
    Maximum precipitation intensity for optical flow computations. Values above 
    R_max are clamped.
  missingval : float
    Value that is used for missing data. No interpolation is done for missing 
    values.
  logtrans : bool
    If True, logarithm is taken from R1 and R2 when computing the motion 
    vectors. This might improve the reliability of motion estimation.
  
  Returns
  -------
  out : array
    List of two-dimensional arrays containing the interpolated precipitation 
    fields ordered by time.
  """
  if R1.shape != R2.shape:
    raise ValueError("R1 and R2 have different shapes")
  
  X,Y = meshgrid(arange(R1.shape[1]), arange(R1.shape[0]))
  W = dstack([X, Y]).astype(float32)
  
  R1_f = _filtered_ubyte_image(R1, R_min, R_max, logtrans=logtrans)
  R2_f = _filtered_ubyte_image(R2, R_min, R_max, logtrans=logtrans)
  
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

def _filtered_ubyte_image(I, R_min, R_max, filter_stddev=1.0, logtrans=False):
  I = I.copy()
  
  MASK = I >= R_min
  I[I > R_max] = R_max
  
  if logtrans == True:
    I = log(I)
    R_min = log(R_min)
    R_max = log(R_max)
  
  I[MASK]  = 128.0 + (I[MASK] - R_min) / (R_max - R_min) * 127.0
  I[~MASK] = 0.0
  
  I = I.astype(ubyte)
  I = gaussian_filter(I, filter_stddev)
  
  return I
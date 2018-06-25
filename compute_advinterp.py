# For given precipitation fields, this script computes a set of temporally 
# interpolated precipitation fields between the time steps. The results are 
# saved in the same format as the input files.

import matplotlib
matplotlib.use('Agg')

from pylab import *
import ConfigParser
from datetime import datetime, timedelta
import os
import shutil
import sys
import driver
import interpolation

config = ConfigParser.RawConfigParser()
config.read("compute_advinterp.cfg")

R_min           = config.getfloat("preprocessing",   "R_min")
R_max           = config.getfloat("preprocessing",   "R_max")
logtrans        = config.getboolean("preprocessing", "logtrans")
output_path     = config.get("output",               "path")
numframes       = config.getint("output",            "numframes")
exporter_method = config.get("output",               "exporter")
save_original   = config.getboolean("output",        "save_original")
farneback_pyr_scale  = config.getfloat("optflow",    "pyr_scale")
farneback_levels     = config.getint("optflow",      "levels")
farneback_winsize    = config.getint("optflow",      "winsize")
farneback_iterations = config.getint("optflow",      "iterations")
farneback_poly_n     = config.getint("optflow",      "poly_n")
farneback_poly_sigma = config.getfloat("optflow",    "poly_sigma")

farneback_params = (farneback_pyr_scale,  farneback_levels, farneback_winsize, 
                    farneback_iterations, farneback_poly_n, farneback_poly_sigma, 0)

def worker(R, geodata, fn, dt, gaugedata, exporter, output_fn_ext, **kwargs):
  print("Interpolating %s <-> %s..." % (str(dt[0]), str(dt[1]))),
  sys.stdout.flush()
  
  R_ip = interpolation.advection(R[0], R[1], numframes, farneback_params, 
                                 R_min=R_min, R_max=R_max, logtrans=logtrans)
  
  print("Done.")
  
  td = timedelta(seconds=(dt[1] - dt[0]).total_seconds()) / (numframes + 1)
  curdate = dt[0] + td
  
  fn_pattern = driver.config_datasource.get("filenames", "pattern")
  fn_ext     = driver.config_datasource.get("filenames", "ext")
  
  for i in xrange(len(R_ip)):
    outfn = datetime.strftime(curdate, fn_pattern) + '.' + fn_ext
    
    print("Saving result to %s...") % outfn,
    sys.stdout.flush()
    exporter(R_ip[i], os.path.join(output_path, outfn), geodata)
    print("Done.")
    
    curdate += td
  
  if save_original == True:
    if output_path != os.path.split(fn[0])[0]:
      shutil.copy(fn[0], output_path)
    if output_path != os.path.split(fn[1])[0]:
      shutil.copy(fn[1], output_path)

driver = driver.Driver()
driver.parse_args()
driver.read_configs()
driver.run(worker, num_prev_precip_fields=1, exporter_method=exporter_method)
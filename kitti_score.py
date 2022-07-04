### this script measures the KITTI submission scores
#   by comparing the results with another top submission
###

import os
import sys
import numpy as np
from PIL import Image

submit_folder = sys.argv[1]
baseline_folder = sys.argv[2]

top_error = []
bottom_error = []
global_error = []
for fn in os.listdir(submit_folder):
    submit_disp = np.array(Image.open(os.path.join(submit_folder, fn)))
    submit_disp = submit_disp.astype(np.float32) / 256.

    baseline_disp = np.array(Image.open(os.path.join(baseline_folder, fn)))
    baseline_disp = baseline_disp.astype(np.float32) / 256.

    top_error.append(np.mean(np.abs(submit_disp[:128, :] - baseline_disp[:128, :])))
    bottom_error.append(np.mean(np.abs(submit_disp[128:, :] - baseline_disp[128:, :])))
    global_error.append(np.mean(np.abs(submit_disp - baseline_disp)))

    print(top_error[-1], bottom_error[-1], global_error[-1])

print('%s vs %s' % (submit_folder, baseline_folder))
print('top error:', np.mean(top_error))
print('bottom error:', np.mean(bottom_error))
print('global error:', np.mean(global_error))

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def property_extraction(design, s_array):
    
    threshold, stroke, normalized_delta_strain, snap, area = 0, 0, 0, 0, 0
    
    L = design['L']
    num_vert = design['num_vert']
    num_hori = design['num_hori']
    normalized_strain = np.linspace(0, 1, 31)
    
    max_idx = argrelextrema(s_array, np.greater)[0]
    min_idx = argrelextrema(s_array, np.less)[0]

    local_max = list(zip(normalized_strain[max_idx], s_array[max_idx]))
    local_min = list(zip(normalized_strain[min_idx], s_array[min_idx]))
    
    # There are some tricky cases that we have local maxima but no local minima.
    if len(max_idx) == 0 or len(min_idx) == 0:
        return local_max, local_min, threshold, stroke, normalized_delta_strain, snap, area

    threshold = local_max[0][1]
    
    threshold_stress = local_max[0][1]
    threshold_strain = local_max[0][0]
    threshold = threshold_stress

    # 2. stroke
    idx_list = []
    for i in range(1, len(s_array)):
        if normalized_strain[i] > threshold_strain: 
            if s_array[i-1] < threshold_stress and s_array[i] >= threshold_stress:
                idx_list.append(i)
    if idx_list:
        idx_last = idx_list[-1]
        normalized_delta_strain = normalized_strain[idx_last] - threshold_strain
    else:
        stroke = 0
    
    L_total = 4.63 * num_vert + num_vert - 1
    nominal_delta_strain = normalized_delta_strain * (0.5*L - 0.35) * num_vert / L_total
    stroke = nominal_delta_strain * L_total

    # 3. snap
    snap = local_max[0][1] - local_min[-1][1]

    # 4. area = stroke * snap
    area = stroke * snap

    return local_max, local_min, threshold, stroke, normalized_delta_strain, snap, area

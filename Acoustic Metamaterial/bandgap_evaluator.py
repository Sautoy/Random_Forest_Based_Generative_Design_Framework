import pandas as pd
import numpy as np

# Calculate stopbands at the target frequency
def targeted_bandgap_calculator(selected_freq, target_freq, threshold, freq_range_of_interests):

    stopbands = band_processing(selected_freq, threshold, freq_range_of_interests)

    target_gap = None
    if stopbands is None:
        return None
    else:
        for gap in stopbands:
            if gap[0] <= target_freq <= gap[1]:
                target_gap = gap
                break    
    return target_gap

# Calculate stopbands and passbands within the selected wave vector range
def bands_calculator(selected_freq, threshold, freq_range_of_interests):

    passranges = []
    for band in range(selected_freq.shape[1]):
        min_freq = np.min(selected_freq[:, band])
        max_freq = np.max(selected_freq[:, band])
        passranges.append([min_freq, max_freq])
    
    passranges = sorted(passranges, key=lambda x: x[0])

    stopbands = []

    first_start, first_end = passranges[0]
    if first_start - freq_range_of_interests[0] > threshold:
        stopbands.append([freq_range_of_interests[0], first_start])

    for i in range(1, len(passranges)):
        prev_end = passranges[i-1][1]
        curr_start = passranges[i][0]
        if curr_start - prev_end > threshold:
            if curr_start < freq_range_of_interests[1]:
                stopbands.append([prev_end, curr_start])
            elif prev_end < freq_range_of_interests[1]:
                stopbands.append([prev_end, freq_range_of_interests[1]])
                break
            else:
                break
    
    last_start, last_end = passranges[-1]
    if freq_range_of_interests[1] - last_end > threshold:
        stopbands.append([last_end, freq_range_of_interests[1]])
    
    passranges = np.array(passranges)
    passranges_list = passranges.tolist()
    passranges_list.sort(key=lambda x: x[0])

    passbands = []
    for start, end in passranges_list:
        if not passbands:
            passbands.append([start, end])
        else:
            prev_start, prev_end = passbands[-1]
            if start <= prev_end:
                passbands[-1][1] = max(prev_end, end)
            else:
                passbands.append([start, end])

    return np.array(stopbands), np.array(passbands)

# Selecting the relevant frequency bands based on the wave vector range
# For the case where target frequency is given, it is for obptimization
# and for the case where target frequency is not given, it is for feasibility check
def band_processing(freq_resize, num_mode, num_k, bg_type, target_freq, threshold, freq_range_of_interests):
    
    if bg_type == -1:  # x
        split_point = int(np.ceil((num_k-1)/3))
        selected_freq = freq_resize[:split_point,:]
    elif bg_type == -2:  # y
        split_point = int(np.ceil((num_k-1)/3))
        selected_freq = freq_resize[split_point+1 : num_k-split_point, :]
    elif bg_type == -3:  # omni-directional
        selected_freq = freq_resize
    else:  
        k_start, k_end = bg_type
        selected_freq = freq_resize[k_start:k_end, :]
    
    if target_freq == -1:
        return bands_calculator(selected_freq, threshold, freq_range_of_interests)
    else:
        return targeted_bandgap_calculator(selected_freq, target_freq, threshold, freq_range_of_interests)

# Calculate the overlap between the target frequency ranges and the stopbands or passbands
# If overlap = 1, it means the target frequency range is fully covered by the stopband or passband
def check_feasibility_from_band_diagram(freq_resize, num_mode, num_k, bg_types, target_freq_ranges, index_stopband, threshold, freq_range_of_interests):
    
    total_target_width = 0.0
    total_overlap = 0.0

    for idx, (bg_type, target_freq_range) in enumerate(zip(bg_types, target_freq_ranges)):
        f_lo, f_hi = target_freq_range
        width = f_hi - f_lo
        total_target_width += width

        stopbands, passbands = band_processing(freq_resize, num_mode, num_k, bg_type, -1, threshold, freq_range_of_interests)

        best_overlap = 0.0

        if idx in index_stopband:
            for b_lo, b_hi in stopbands:
                inter = max(0.0, min(f_hi, b_hi) - max(f_lo, b_lo))
                best_overlap = max(best_overlap, inter)
        else:
            for b_lo, b_hi in passbands:
                inter = max(0.0, min(f_hi, b_hi) - max(f_lo, b_lo))
                best_overlap = max(best_overlap, inter)

        total_overlap += best_overlap

    return total_overlap / total_target_width if total_target_width > 0 else 0.0


def objective_function_from_list(gap_result, target_freqs, index_stopband):
    
    min_dfreq = []
        
    for idx, target_freq in enumerate(target_freqs):
        if idx in index_stopband:
            if gap_result is not None:
                flag = 0
                for gap in gap_result:
                    if gap[0] <= target_freq <= gap[1]:
                        distance = min((gap[1] - target_freq), (target_freq - gap[0]))
                        min_dfreq.append(distance)
                        flag = 1
                        break
                if flag == 0:
                    return 0.0
            else:
                return 0.0
        else:
            if gap_result is not None:
                for gap in gap_result:
                    if gap[0] <= target_freq <= gap[1]:
                        return 0.0
    return min(min_dfreq)


def objective_function_from_band_diagram(freq_resize, num_mode, num_k, bg_types, target_freqs, index_stopband, threshold, freq_range_of_interests):

    min_dfreq = []
        
    for idx, (bg_type, target_freq) in enumerate(zip(bg_types, target_freqs)):
        gap_result = band_processing(freq_resize, num_mode, num_k, bg_type, target_freq, threshold, freq_range_of_interests)

        if idx in index_stopband:
            if gap_result is not None and len(gap_result) == 2:
                distance = min((gap_result[1] - target_freq), (target_freq - gap_result[0]))
                min_dfreq.append(distance)
            else:
                return 0.0
        else:
            if gap_result is not None and len(gap_result) == 2:
                return 0.0
    if not index_stopband:
        return 1.0

    return min(min_dfreq) if min_dfreq else 0.0
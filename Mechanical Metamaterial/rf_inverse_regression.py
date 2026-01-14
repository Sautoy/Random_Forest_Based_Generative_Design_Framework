import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from joblib import load
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import plotly.graph_objects as go
import os
from inverse import *
from forward import predict_frequencies
from rf_forward_k import get_bandgap_ranges
from utils import create_dir

import sys
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)

EPSILON = 1e-6

# if __name__ == "__main__":
    
#     # Load configuration
#     current_dir = os.getcwd() 
#     config_file_path = os.path.join(current_dir, "RIGID_K_o3d/config_k.json")
#     print(config_file_path)

#     config_file_path = config_file_path.replace("\\", "/")

#     with open(config_file_path, 'r') as file:
#         config = json.load(file)

#     # Load the forest, frequency intervals, and design variable bounds
#     clf = load('results_rf_k/forest.joblib')
#     fre_intervals = np.load('results_rf_k/fre_intervals.npy')
#     k_intervals = np.load('results_rf_k/k_intervals.npy')
#     des_var_bounds = np.load('results_rf_k/des_var_bounds.npy')

#     # Define target bandgap
#     stopband = config['stopband']
#     passband = config['passband']
#     print('stopband:', stopband)
#     print('passband:', passband)

#     # Check data satisfaction (based on RIGID data set)
#     num_mode = config['num_mode']
#     num_k = config['num_k']
#     raw_data = pd.read_csv(config['data_path'], sep=',')
#     bandgaps_data = get_bandgap_ranges(raw_data, num_mode)
#     # recalls, _ = evaluate_satisfaction(target_bandgap, index_stopband, bandgaps_data)
#     # print('{}/{} feasible designs in data.'.format(sum(recalls==1), len(recalls))) # Only ratio = 1.0 is completely feasible
    
#     # Features
#     des_var_names = config['design_variable_names']
#     n_des_vars = len(des_var_names)
    
#     # Get trees from the forest
#     estimators = clf.estimators_
#     n_estimators = len(estimators)
    
#     print('Getting feasible regions for all trees ...')
#     list_des_var_ranges = []
#     valid_des_var_ranges = []
#     for i, dt in enumerate(tqdm(estimators)):

#         # Get the design variable ranges that satisfy the target bandgap
#         [des_var_ranges, skip] = inverse_design(dt, passband, stopband, fre_intervals[:,1], k_intervals[:,1], des_var_bounds, verbose=0)
#         list_des_var_ranges.append(des_var_ranges)
#         valid_des_var_ranges.append(skip)
    
#     list_des_var_ranges_filtered = [
#         element for element, valid in zip(list_des_var_ranges, valid_des_var_ranges) if not valid
#     ]
    
#     print(len(list_des_var_ranges_filtered), 'trees are valid.')

#     # Generate designs from the obtained design variable ranges
#     print('Generating designs ...')
#     n_designs = config['num_generated_designs'] # Number of designs to generate
#     cat_var_indices = config['categorical_variable_indices']
#     sample_threshold = config['sample_threshold']

#     # designs, probabilities = generate(n_designs, list_des_var_ranges, des_var_bounds, cat_var_indices)
#     designs, probabilities = generate(n_designs, list_des_var_ranges_filtered, des_var_bounds, cat_var_indices)

#     stopband_str = "stopband_" + "_".join([f"k:{rng[0]}-{rng[1]},freq:{rng[2]}-{rng[3]}" for rng in stopband])
#     passband_str = "passband_" + "_".join([f"k:{rng[0]}-{rng[1]},freq:{rng[2]}-{rng[3]}" for rng in passband])

#     if len(designs) > 0:
        
#         # Create a folder for this experiment
#         exp_dir = 'results_rf/exp_' + stopband_str + passband_str
#         create_dir(exp_dir)
#         # with open('{}/config.json'.format(exp_dir), 'w') as fout:
#         #     config['target_bandgap'] = target_bandgap
#         #     json.dump(config, fout)
        
#         # Save feasible design variable ranges for all the estimators
#         # with open('{}/list_des_var_ranges.json'.format(exp_dir), 'w') as fout:
#         #     json.dump(list_des_var_ranges_filtered, fout)

#         # Convert to serializable dictionaries
#         serializable_meshes = [
#             mesh.to_dict() for mesh in list_des_var_ranges_filtered
#         ]

#         # Save to JSON file
#         with open(f"{exp_dir}/list_des_var_ranges.json", "w") as fout:
#             json.dump(serializable_meshes, fout)
            
#         # Save designs and their probabilities
#         gen_des_df = pd.DataFrame(designs, columns=des_var_names)
#         gen_des_df['Probability'] = probabilities
#         gen_des_path = '{}/generated_designs.csv'.format(exp_dir)
#         gen_des_df.to_csv(gen_des_path, index=False)
        
#         # Plot predicted bandgaps of generated designs
#         # indicators_pred, indicators_pred_proba = predict_frequencies(clf, designs, fre_intervals[:,1])
        
#         # Get designs from data with satisfied target bandgaps
#         # satisfactions = (recalls == 1)
#         # designs_data = raw_data[des_var_names].to_numpy()
#         # designs_data = np.nan_to_num(designs_data)
#         # satisfied_designs_data = designs_data[satisfactions]
#         # unsatisfied_designs_data = designs_data[np.logical_not(satisfactions)]
        
#         des_x, des_y, des_z, prob_val = discretize_des_space(des_var_bounds, list_des_var_ranges_filtered, num_x = 10)
        

#         fig = go.Figure(data=go.Volume(
#             x=des_x,
#             y=des_y,
#             z=des_z,
#             value = prob_val,        
#             isomin = prob_val.min(),                # Minimum value to render
#             isomax = prob_val.max(),                 # Maximum value to render
#             opacity = 0.6,                # Opacity (lower values = more transparent)
#             opacityscale=[
#             [0, 0],   # val = 0 ➝ opacity = 0 (transparent)
#             [0.5, 0.25], # val = 0.5 ➝ opacity = 0.5
#             [0.75, 0.5], # val = 0.75 ➝ opacity = 0.75
#             [1, 1]    # val = 1 ➝ opacity = 1 (opaque)
#             ],             # Opacity (lower values = more transparent)
#             surface_count = 20,           # Number of transparent isosurfaces
#             colorscale="Reds",       # Heatmap color scale (choose from Plotly's color scales)
#         ))

#         # fig.add_trace(go.Scatter3d(x=satisfied_designs_data[:,0], y=satisfied_designs_data[:,1], z=satisfied_designs_data[:,2],
#         #                             mode='markers', marker=dict(size=4, color='green', opacity=0.8),showlegend=False))
#         # fig.add_trace(go.Scatter3d(x=unsatisfied_designs_data[:,0], y=unsatisfied_designs_data[:,1], z=unsatisfied_designs_data[:,2],
#         #                             mode='markers', marker=dict(size=4, color='grey', opacity=0.8),showlegend=False))
#         fig.add_trace(go.Scatter3d(x=designs[:,0], y=designs[:,1], z=designs[:,2],
#                             mode='markers', marker=dict(size=4, color='lightgreen', opacity=0.8),showlegend=False))

#         # Update layout
#         fig.update_layout(
#             title=f"{stopband_str}; {passband_str}",
#             autosize=False,
#             width=800, 
#             height=800,
#             margin=dict(l=65, r=50, b=65, t=90),
#             scene=dict(
#             xaxis_title=des_var_names[0],
#             yaxis_title=des_var_names[1],
#             zaxis_title=des_var_names[2]
#         ))


#         # Save the plot
#         plot_path = '{}/design_space.html'.format(exp_dir)
#         fig.write_html(plot_path)
    
#     else:
#         print('Cannot find designs that meet the target!')
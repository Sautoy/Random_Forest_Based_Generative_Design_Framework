import json
from pathlib import Path
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
from utils import create_dir
import re
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from bandgap_evaluator import check_feasibility_from_band_diagram
from inverse import calculate_proba_in_des_space_forward_regression, convert_target_format, random_target_2d

import sys
np.set_printoptions(threshold=sys.maxsize, edgeitems=30, linewidth=100000, precision=2)

EPSILON = 1e-6

if __name__ == "__main__":

    # Random target setting
    # with open("config_regression.json", "r") as f:
    #     config = json.load(f)
    
    # target = random_target_2d(0, 3, 20.0, 50.0, 1.0, 3.0)
    
    # config["stopband"] = target["stopband"]
    # config["passband"] = target["passband"]
    
    # with open("config_regression.json", "w") as f:
    #     json.dump(config, f, indent=4)


    # Load configuration
    with open("config_regression.json", "r") as f:
        config = json.load(f)

    reg = load('results_rf_regression/forest.joblib')
    des_var_bounds = np.load('results_rf_regression/des_var_bounds.npy')

    stopband = config['stopband']
    passband = config['passband']
    print('stopband:', stopband)
    print('passband:', passband)

    num_mode = config['num_mode']
    num_k = config['num_k']
    raw_data = pd.read_csv(config['data_path'], sep=',')

    des_var_names = config['design_variable_names']
    n_des_vars = len(des_var_names)

    estimators = reg.estimators_
    n_estimators = len(estimators)


    # Load information from the dataset
    freq_range_of_interests = config['freq_of_interest']
    threshold = 1
    
    folder_path = r'D:\My Drive\Research\WAM\RIGID_2DAM_reg\dataset\lhs_consider_constrain1_n_samples_500_num_k_61_num_mode_15_seed_33'
    designs = []
    band_structures = []
    for file_name in os.listdir(folder_path):
        match = re.match(r'result_rstr_([\d.]+)_rcenter_([\d.]+)_rcorner_([\d.]+)\.csv', file_name)
        if match:  # Check if file matches the required pattern
            rstr = float(match.group(1))
            rcenter = float(match.group(2))
            rcorner = float(match.group(3))
            designs.append({'rstr': rstr, 'rcenter': rcenter, 'rcorner': rcorner})
            
        file_path = os.path.join(folder_path, file_name)
        csv_data = pd.read_csv(file_path, skiprows = 4) 
        csv_data['Eigenfrequency (kHz)'] = csv_data['Eigenfrequency (kHz)'].apply(
            lambda x: complex(x.replace('i', 'j'))) 
        band_structures.append(csv_data)
    

    # Check feasible designs in the dataset
    bg_type, target_freq_range, index_stopband = convert_target_format(stopband, passband, num_k)

    feasibility_list = []
    k_stride = (num_k - 1)/3.0

    for i, design in enumerate(designs):
        data = band_structures[i]
        data['Frequency_real'] = data['Eigenfrequency (kHz)'].apply(lambda x: np.real(x))
        freq = data['Frequency_real'].values.reshape(num_k, num_mode)
        feasibility_val = check_feasibility_from_band_diagram(freq, num_mode, num_k, bg_type, 
                                    target_freq_range, index_stopband, threshold, freq_range_of_interests)
        feasibility_list.append({
            'design': design,
            'feasibility value': feasibility_val
        })

        freq = data['Frequency_real'].values.reshape(num_k, num_mode)
        k_points = data['% k'].values.reshape(num_k, num_mode)[:, 0]

    feasible_indices = [i for i, item in sorted(enumerate(feasibility_list), 
                        key=lambda x: x[1]['feasibility value'], 
                        reverse=True) 
                    if item['feasibility value'] == 1]

    print(f"Number of feasible designs: {len(feasible_indices)}")

    property_distribution = []
    property_distribution_remap = []

    for item in feasibility_list:
        design = item['design']
        feasibility_val = item['feasibility value']
        property_distribution.append({
            'rstr': design['rstr'],
            'rcenter': design['rcenter'],
            'rcorner': design['rcorner'],
            'Feasibility Value': feasibility_val,
        })

        rstr_val = design['rstr']
        rcenter_val = design['rcenter']
        rcorner_val = design['rcorner']

    df_property_distribution = pd.DataFrame(property_distribution)

    min_size = 5
    size_rate = 10

    df_property_distribution['Point Size'] = np.where(
        df_property_distribution['Feasibility Value'] == 0,
        min_size,
        min_size + df_property_distribution['Feasibility Value'] * size_rate
    )


    # Calculate likelihood distribution in the design space
    des_x, des_y, des_z, prob_forward = calculate_proba_in_des_space_forward_regression(des_var_bounds, passband, stopband, num_mode, num_k, reg, 10, 1)

    # Generate designs from the obtained design variable ranges
    stopband_str = "stopband_" + "_".join([f"k_{rng[0]}_{rng[1]}_freq_{rng[2]}_{rng[3]}" for rng in stopband])
    passband_str = "passband_" + "_".join([f"k_{rng[0]}_{rng[1]}_freq_{rng[2]}_{rng[3]}" for rng in passband])

    i = np.argmax(prob_forward)                              
    initial_design = np.array([des_x[i], des_y[i], des_z[i]]) # Using the maximum likelihood design as the initial design 

    print('Generating designs ...')
    n_designs = config['num_generated_designs'] 
    cat_var_indices = config['categorical_variable_indices']
    generated_designs, probabilities = generate_forward_regression(n_designs, passband, stopband, num_mode, num_k, des_var_bounds, reg, cat_var_indices, 1, initial_design, 0.2)
    


    # Store and visualize results
    if len(generated_designs) > 0:
        
        exp_dir = 'results_rf_regression/exp_' + stopband_str + passband_str
        create_dir(exp_dir)
        
        gen_des_df = pd.DataFrame(generated_designs, columns=des_var_names)
        gen_des_df['Probability'] = probabilities
        gen_des_path = '{}/generated_designs.csv'.format(exp_dir)
        gen_des_df.to_csv(gen_des_path, index=False)

    exp_dir       = 'results_rf_regression/exp_' + stopband_str + passband_str
    gen_des_path  = f'{exp_dir}/generated_designs.csv'

    gen_des_df = pd.read_csv(gen_des_path)

    generated_designs = gen_des_df[des_var_names].values    # shape (N, len(des_var_names))
    probabilities     = gen_des_df['Probability'].values    # shape (N,)

    fig = go.Figure(data=go.Volume(
        x=des_x,
        y=des_y,
        z=des_z,
        value = prob_forward,        
        isomin = prob_forward.min(),                # Minimum value to render
        isomax = prob_forward.max(),                # Maximum value to render
        opacity = 0.6,                # Opacity (lower values = more transparent)
        opacityscale=[
        [0, 0],        # val = 0 ➝ opacity = 0 (transparent)
        [0.5, 0.25],   # val = 0.5 ➝ opacity = 0.5
        [0.75, 0.5],   # val = 0.75 ➝ opacity = 0.75
        [1, 1]         # val = 1 ➝ opacity = 1 (opaque)
        ],             # Opacity (lower values = more transparent)
        surface_count = 20,      # Number of transparent isosurfaces
        colorscale="Reds",       # Heatmap color scale (choose from Plotly's color scales)
    ))

    fig.add_trace(go.Scatter3d(
        x=gen_des_df["Strut Radius"], 
        y=gen_des_df["Center Mass Radius"],
        z=gen_des_df["Corner Mass Radius"], 
        mode='markers', 
        marker=dict(
            size=gen_des_df['Probability'] * 30,  # Adjust size based on probability
            color='blue',  # Color for generated designs
            opacity=0.5),
            showlegend=False))
    
    fig.add_trace(go.Scatter3d(
        x=df_property_distribution['rstr'],
        y=df_property_distribution['rcenter'],
        z=df_property_distribution['rcorner'],
        mode='markers',
        marker=dict(
            size=df_property_distribution['Point Size'],
            color=np.where(
                df_property_distribution['Feasibility Value'] == 1,  # Condition for coloring
                'green',
                'gray'
            ),
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name='Design Points',
        hoverinfo='text',
        hovertext='Objective Value: ' + df_property_distribution['Feasibility Value'].astype(str)
    ))

    fig.update_layout(
        title=f"{stopband_str}; {passband_str}",
        autosize=False,
        width=800, 
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
        xaxis_title=des_var_names[0],
        yaxis_title=des_var_names[1],
        zaxis_title=des_var_names[2]
    ))

    exp_dir = Path(exp_dir)

    output_path = exp_dir / "generate3d.html"

    fig.write_html(str(output_path), include_plotlyjs="cdn")


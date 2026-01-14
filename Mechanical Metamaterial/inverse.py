from tqdm import tqdm
import numpy as np
from sklearn.tree import _tree
from itertools import product
from scipy.signal import argrelextrema

EPSILON = 1e-8


def uniform_designs(des_var_bounds, cat_var_indices, n_designs = 20):
    n_des_vars = des_var_bounds.shape[0]
    con_var_indices = [i for i in range(n_des_vars) if i not in cat_var_indices]
    designs = np.zeros((n_designs, n_des_vars))
    for i in cat_var_indices:
        designs[:, i] = np.random.choice(np.arange(des_var_bounds[i, 0], des_var_bounds[i, 1]+1), size=n_designs)
    designs[:, con_var_indices] = np.random.uniform(des_var_bounds[con_var_indices, 0], des_var_bounds[con_var_indices, 1],
                                                    size=(n_designs, len(con_var_indices)))
    return np.squeeze(designs)

def uniform_designs_defined(des_var_bounds, cat_var_indices, n_designs = 50, max_attempts = 5000):
    n_des_vars = des_var_bounds.shape[0]
    con_var_indices = [i for i in range(n_des_vars) if i not in cat_var_indices]
    
    accepted = []
    attempts = 0

    while len(accepted) < n_designs and attempts < max_attempts:
        design = np.zeros(n_des_vars)

        # Step 1: Sample categorical variables
        for i in cat_var_indices:
            design[i] = np.random.choice(np.arange(des_var_bounds[i, 0], des_var_bounds[i, 1] + 1))

        # Step 2: Sample continuous variables within loose bounding box
        for i in con_var_indices:
            design[i] = np.random.uniform(des_var_bounds[i, 0], des_var_bounds[i, 1])

        # Step 3: Check if the design satisfies nonlinear/dependent constraints
        if is_in_defined_bounds(design):
            accepted.append(design)

        attempts += 1

    if len(accepted) < n_designs:
        raise RuntimeError(f"Only {len(accepted)} valid designs found after {max_attempts} attempts.")

    return np.array(accepted)


def check_feasibility_for_target(design, response, target):
    
    threshold, stroke, normalized_delta_strain, snap, area = 0, 0, 0, 0, 0
    
    L = design[0]
    num_vert = design[3]
    num_hori = design[4]
    normalized_strain = np.linspace(0, 1, 31)
    
    max_idx = argrelextrema(response, np.greater)[0]
    min_idx = argrelextrema(response, np.less)[0]

    local_max = list(zip(normalized_strain[max_idx], response[max_idx]))
    local_min = list(zip(normalized_strain[min_idx], response[min_idx]))

    # There are some tricky cases that we have local maxima but no local minima.
    if len(max_idx) == 0 or len(min_idx) == 0:
        if np.isclose(stroke, target['stroke'], atol= target['stroke']*target['tol_stroke']) and np.isclose(threshold, target['threshold'], atol=target['threshold']*target['tol_threshold']):
            return 1
        else:
            return 0

    threshold = local_max[0][1]
    
    threshold_stress = local_max[0][1]
    threshold_strain = local_max[0][0]
    threshold = threshold_stress

    # 2. stroke
    idx_list = []
    for i in range(1, len(response)):
        if normalized_strain[i] > threshold_strain: 
            if response[i-1] < threshold_stress and response[i] >= threshold_stress:
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

    if np.isclose(stroke, target['stroke'], atol= target['stroke']*target['tol_stroke']) and np.isclose(threshold, target['threshold'], atol=target['threshold']*target['tol_threshold']):
        return 1
    else:
        return 0


def evaluate_proba(design, target, reg):
    
    eps_n = np.linspace(0, 1, 31)
    eps_inp = eps_n[1:]

    x_array = np.hstack([np.tile(design, (len(eps_inp), 1)), eps_inp.reshape(-1, 1)])

    match_list = []

    for tree in reg.estimators_:

        response = tree.predict(x_array)
        
        match = check_feasibility_for_target(design, response, target)
        match_list.append(match)
    
    match_list = np.array(match_list)
    proba = np.sum(match_list == 1.0) / len(match_list)

    return proba


def is_in_bounds(design, des_var_bounds):
    flag = np.logical_and(design >= des_var_bounds[:, 0], design <= des_var_bounds[:, 1])
    return np.all(flag)

def is_in_defined_bounds(design):
    rstrut, rcenter, rcorner = design[:3]  
    
    if not (4 <= rstrut <= 6.41):
        return False
    if not (np.sqrt(2) * rstrut <= rcenter <= 20):
        return False
    if not ((np.sqrt(2) + 1) * rstrut <= rcorner <= 20):
        return False

    return True

def mcmc_updater(curr_state, curr_likeli,  likelihood, cat_var_indices, des_var_bounds, defined = 0):
    """ 
    Reference: 
        https://exowanderer.medium.com/metropolis-hastings-mcmc-from-scratch-in-python-c21e53c485b7
        
    Propose a new state and compare the likelihoods
    
    Given the current state (initially random), 
      current likelihood, the likelihood function, and 
      the transition (proposal) distribution, `mcmc_updater` generates 
      a new proposal, evaluate its likelihood, compares that to the current 
      likelihood with a uniformly samples threshold, 
    then it returns new or current state in the MCMC chain.

    Args:
        curr_state (float): the current parameter/state value
        curr_likeli (float): the current likelihood estimate
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the 
          next proposal state

    Returns:
        (tuple): either the current state or the new state
          and its corresponding likelihood
    """
    # Set step size (Reference: Automatic Step Size Selection in Random Walk Metropolis Algorithms, Todd L. Graves, 2011.)
    n_des_vars = des_var_bounds.shape[0]
    con_var_indices = [i for i in range(n_des_vars) if i not in cat_var_indices]
    proposal_state = curr_state.copy()
    stepsize = 1.4 * len(con_var_indices)**(-0.5) * (des_var_bounds[con_var_indices,1] - des_var_bounds[con_var_indices,0])
    # Generate a proposal state using the proposal distribution (using normal distribution)
    # Proposal state == new guess state to be compared to current
    proposal_state[con_var_indices] = np.random.normal(curr_state[con_var_indices], stepsize)
    if len(cat_var_indices) > 0:
        rand_cat_idx = np.random.choice(cat_var_indices)
        proposal_state[rand_cat_idx] = np.random.choice(np.arange(des_var_bounds[rand_cat_idx, 0], des_var_bounds[rand_cat_idx, 1]+1))

    # Calculate the acceptance criterion
    prop_likeli = likelihood(proposal_state)
    accept_crit = prop_likeli / (curr_likeli + EPSILON)

    # Generate a random number between 0 and 1
    accept_threshold = np.random.uniform(0, 1)

    # If the acceptance criterion is greater than the random number,
    # accept the proposal state as the current state
    # if accept_crit > accept_threshold and is_in_bounds(proposal_state, des_var_bounds):
    #     return proposal_state, prop_likeli, 1

    if defined:
        if accept_crit > accept_threshold and is_in_defined_bounds(proposal_state):
            return proposal_state, prop_likeli, 1
        else:
            return curr_state, curr_likeli, 0
    else:
        if accept_crit > accept_threshold and is_in_bounds(proposal_state, des_var_bounds):
            return proposal_state, prop_likeli, 1
        else:
            return curr_state, curr_likeli, 0

def generate_forward(n_designs, passband, stopband, fre_intervals, k_intervals, des_var_bounds, clf, cat_var_indices, defined = 0,
                    initial_design = None, burnin = 0.2):
    
    n_des_vars = des_var_bounds.shape[0]

    if initial_design is None:    
        print('Deciding the initial design for MCMC ...')
        
        # Uniformly sample designs within the design variable bounds
        if defined:
            # Generate designs within the defined bounds
            designs = uniform_designs_defined(des_var_bounds, cat_var_indices, n_designs= 25 * n_des_vars)
            # designs = uniform_designs_defined(des_var_bounds, cat_var_indices, 100)
        else:
            designs = uniform_designs(des_var_bounds, cat_var_indices, n_designs= 15 * n_des_vars)
        
        # Select the one with the highest likelihood as the initial design
        max_proba = 0
        for design in tqdm(designs):
            proba = evaluate_proba(passband, stopband, design, fre_intervals, k_intervals, des_var_bounds, clf)
            if proba >= max_proba:
                initial_design = design
                max_proba = proba # I add this line
    
    print('Generating designs using MCMC ...')
    # The number of samples in the burn in phase
    # n_designs: Number of designs to generate
    n_burnin = int(burnin * n_designs)
    # Set the current state to the initial state
    curr_state = initial_design
    curr_likeli = evaluate_proba(passband, stopband, curr_state, fre_intervals, k_intervals, des_var_bounds, clf)
    
    # Metropolis-Hastings with unique samples
    designs = []
    probabilities = []
    count = 0
    accepted_count = 0
    with tqdm(total=n_designs) as pbar:
        while accepted_count < n_designs:
            # The proposal distribution sampling and comparison
            # occur within the mcmc_updater routine
            curr_state, curr_likeli, flag = mcmc_updater(
                curr_state=curr_state,
                curr_likeli=curr_likeli,
                likelihood=lambda x: evaluate_proba(passband, stopband, x, fre_intervals, k_intervals, des_var_bounds, clf),
                cat_var_indices=cat_var_indices,
                des_var_bounds=des_var_bounds,
                defined=defined
            )
            count += 1
    
            # Append the current state to the list of samples
            # Flag = 1: the proposal state is accepted 
            if count > n_burnin and flag:
                # Only append after the burnin to avoid including
                # parts of the chain that are prior-dominated
                designs.append(curr_state)
                probabilities.append(curr_likeli)
                accepted_count += 1
                pbar.update(1)
    return np.array(designs), np.array(probabilities)

def calculate_proba_in_des_space(des_var_bounds, target, reg, num_d = 20, defined = 0):

    x = np.linspace(des_var_bounds[0][0], des_var_bounds[0][1], num_d)  # L
    y = np.linspace(des_var_bounds[1][0], des_var_bounds[1][1], num_d)  # w
    z = np.linspace(des_var_bounds[2][0], des_var_bounds[2][1], num_d)  # alpha
    v = np.linspace(des_var_bounds[3][0], des_var_bounds[3][1], 4)      # num_vert
    h = np.linspace(des_var_bounds[4][0], des_var_bounds[4][1], 4)      # num_hori

    grids = np.meshgrid(x, y, z, v, h, indexing='ij') 
    flat_grids = [g.flatten() for g in grids]
    points = np.stack(flat_grids, axis=1)

    prob_val = np.zeros(len(points))
    
    for i, pt in enumerate(points):
        if defined:
            if is_in_defined_bounds(pt):
                prob_val[i] = evaluate_proba(pt, target, reg)
            else:
                prob_val[i] = 0
        else:
            prob_val[i] = evaluate_proba(pt, target, reg)

    return grids, prob_val
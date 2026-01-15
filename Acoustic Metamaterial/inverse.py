from tqdm import tqdm
from functools import reduce
import numpy as np
from sklearn.tree import _tree
from bandgap_evaluator import check_feasibility_from_band_diagram
from itertools import product

EPSILON = 1e-8

def random_target_2d(k_lb, k_ub, fre_lb, fre_ub, min_width, max_width, resolution = 1):
    
    """
    Returns a dict with two keys, 'stopband' and 'passband', each
    a list of [k_lo, k_hi, f_lo, f_hi] ranges.

    - You get either 2 or 3 total ranges.
    - k_lo, k_hi are ints in [k_lb, k_ub], sorted.
    - f_lo, f_hi are floats in [fre_lb, fre_ub],
      with (f_hi - f_lo) in (min_width, max_width).
    - Frequency ranges do not overlap.
    - Randomly assign 1–2 of them to stopband, rest to passband.
    """

    n_total = np.random.choice([2, 3])

    segments = []
    while len(segments) < n_total:
        # pick integer k‐range
        ks = np.random.choice(np.arange(k_lb, k_ub + 1),
                              size=2, replace=False)
        k_lo, k_hi = np.sort(ks)

        # pick non-overlapping frequency range
        while True:
            fs = np.random.uniform(fre_lb, fre_ub, size=2)
            f_lo, f_hi = np.sort(fs)
            width = f_hi - f_lo
            if not (min_width < width < max_width):
                continue
            overlap = False
            for _, _, existing_lo, existing_hi in segments:
                if not (f_hi <= existing_lo or f_lo >= existing_hi):
                    overlap = True
                    break
            if not overlap:
                break

        # round to 1 decimals
        f_lo = round(f_lo, resolution)
        f_hi = round(f_hi, resolution)

        segments.append([int(k_lo), int(k_hi), float(f_lo), float(f_hi)])

    max_stop = min(2, n_total)
    n_stop = np.random.choice(np.arange(1, max_stop + 1))
    indices = np.arange(n_total)
    stop_idx = np.random.choice(indices, size=n_stop, replace=False)
    
    stopband = [segments[i] for i in stop_idx]
    passband = [segments[i] for i in indices if i not in stop_idx]

    return {"stopband": stopband, "passband": passband}


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


def evaluate_proba_forward_regression(passband, stopband, design, num_mode, num_k, reg):
    
    bg_types, target_freq_range, index_stopband = convert_target_format(stopband, passband, num_k)

    k_cart =  np.linspace(0, 3, num = num_k)
    mode_cart = np.arange(1, num_mode + 1)

    combinations = np.array(list(product(k_cart, mode_cart)))  # shape: (len(k)*len(fre), 2)
    design_repeated = np.tile(design, (combinations.shape[0], 1))  # shape: (N, 3)
    x_array = np.hstack((design_repeated, combinations))

    match_list = []

    for tree in reg.estimators_:

        response = tree.predict(x_array)
        freq_resize = response.reshape(num_k, num_mode)
        
        match = check_feasibility_from_band_diagram(freq_resize, num_mode, num_k, bg_types, target_freq_range, index_stopband, 1, [0, 60])
        match_list.append(match)
    
    match_list = np.array(match_list)
    proba = np.sum(match_list == 1.0) / len(match_list)

    return proba

def convert_target_format(stopband, passband, num_k):
    k_vals = np.linspace(0, 3, num=num_k)

    bg_types = []
    target_freq_range = []
    index_stopband = []

    all_bands = stopband + passband
    total = len(all_bands)

    for i, band in enumerate(all_bands):
        k_start_val, k_end_val, f_start, f_end = band

        k_start_idx = np.argmin(np.abs(k_vals - k_start_val))
        k_end_idx = np.argmin(np.abs(k_vals - k_end_val))

        k_idx_range = sorted([k_start_idx, k_end_idx])

        bg_types.append(k_idx_range)
        target_freq_range.append([f_start, f_end])

        if i < len(stopband):  
            index_stopband.append(i)

    return bg_types, target_freq_range, index_stopband

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
    stepsize = 1.5 * len(con_var_indices)**(-0.5) * (des_var_bounds[con_var_indices,1] - des_var_bounds[con_var_indices,0])
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


def generate_forward_regression(n_designs, passband, stopband, num_mode, num_k, des_var_bounds, reg, cat_var_indices, defined = 0,
                                initial_design = None, burnin = 0.2, max_iters = 2000):
    
    terminate = True
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
            proba = evaluate_proba_forward_regression(passband, stopband, design, num_mode, num_k, reg)
            if proba >= max_proba:
                initial_design = design
                max_proba = proba
    
    print('Generating designs using MCMC ...')
    # The number of samples in the burn in phase
    # n_designs: Number of designs to generate
    n_burnin = int(burnin * n_designs)
    # Set the current state to the initial state
    curr_state = initial_design
    curr_likeli = evaluate_proba_forward_regression(passband, stopband, curr_state, num_mode, num_k, reg)
    
    if curr_likeli > 0:
        terminate = False

    # Metropolis-Hastings with unique samples
    designs = []
    probabilities = []
    count = 0
    accepted_count = 0
    accept_anyway = False

    with tqdm(total=n_designs) as pbar:
        while accepted_count < n_designs:
            # The proposal distribution sampling and comparison
            # occur within the mcmc_updater routine
            curr_state, curr_likeli, flag = mcmc_updater(
                curr_state=curr_state,
                curr_likeli=curr_likeli,
                likelihood=lambda x: evaluate_proba_forward_regression(passband, stopband, x, num_mode, num_k, reg),
                cat_var_indices=cat_var_indices,
                des_var_bounds=des_var_bounds,
                defined=defined
            )
            count += 1
    
            if terminate:
                if count >= max_iters and accepted_count == 0:
                    accept_anyway = True  
                    print(f'Warning: No designs accepted after {max_iters} iterations. Accepting anyway.')  

                if count > n_burnin:
                    if flag or accept_anyway:
                        designs.append(curr_state)
                        probabilities.append(curr_likeli)
                        accepted_count += 1
                        pbar.update(1)
            else:
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



def calculate_proba_in_des_space_forward_regression(des_var_bounds, passband, stopband, num_mode, num_k, reg, num_x = 20, defined = 0):

    x = np.linspace(des_var_bounds[0][0], des_var_bounds[0][1], num_x)
    y = np.linspace(des_var_bounds[1][0], des_var_bounds[1][1], num_x)
    z = np.linspace(des_var_bounds[2][0], des_var_bounds[2][1], num_x)
    X, Y, Z = np.meshgrid(x, y, z)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    points = np.vstack([x_flat, y_flat, z_flat]).T
    prob_val = np.zeros(len(x_flat))
    
    for i, pt in enumerate(points):
        if defined:
            if is_in_defined_bounds(pt):
                prob_val[i] = evaluate_proba_forward_regression(passband, stopband, pt, num_mode, num_k, reg)
            else:
                prob_val[i] = 0
        else:
            prob_val[i] = evaluate_proba_forward_regression(passband, stopband, pt, num_mode, num_k, reg)

    return x_flat, y_flat, z_flat, prob_val
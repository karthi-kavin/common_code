"""
New Parameters:
  season_radius: determines range to change weighting for, 7 is good starting place (assumes season is 7*2+1 = 15 weeks). Should research season of each disease.

  weight_boost: Sets the scaling factor for weights from past or future years that fall in the same seasonal window (determined by season_radius).
  A value of 1 corresponds to no boosting.
    
  weight_type='flat'  # 'flat', 'triangular', or 'gaussian'
  'flat': applies the same weight_boost factor to all points within the season_radius.

  'triangular': linearly decreases weight with distance from the seasonal center.

  'gaussian': applies a Gaussian-like weighting centered around the seasonal phase, with standard deviation = season_radius / 2. (flatter near center, falls off quickly at edge, middle ground between flat and triangular)

"""

import numpy as np
from scipy import integrate
import scipy
from scipy import ndimage
from scipy import stats

# @title Modified sliding median
def seasonal_weighted_median_filter(
    signal,
    window_size=101,
    season_length=52,
    season_radius=0,
    weight_boost=1,
    weight_type='flat'  # 'flat', 'triangular', or 'gaussian'
):
    signal = np.asarray(signal)
    N = len(signal)

    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    padded = np.pad(signal, (half, half), mode='symmetric')
    result = np.empty(N)

    for i in range(N):
        window = padded[i : i + window_size]

        # Signal-relative indices
        signal_indices = np.arange(i, i + window_size) - half

        # Week + Year info
        mod_indices = signal_indices % season_length
        years = signal_indices // season_length
        center_mod = i % season_length
        center_year = i // season_length

        dists = np.abs(mod_indices - center_mod)
        dists = np.minimum(dists, season_length - dists)

        weights = np.ones_like(window, dtype=float)

        # Apply boosted weights only to other years in same seasonal phase
        boost_mask = (dists <= season_radius) & (years != center_year)
        if weight_type == 'flat':
            weights[boost_mask] *= weight_boost

        elif weight_type == 'triangular':
            weights[boost_mask] *= (1 + (season_radius - dists[boost_mask]) / season_radius) * weight_boost

        elif weight_type == 'gaussian':
            sigma = season_radius / 2.0
            weights[boost_mask] *= (1 + np.exp(-0.5 * (dists[boost_mask] / sigma) ** 2)) * weight_boost

        # Weighted median
        sorted_idx = np.argsort(window)
        sorted_values = window[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cum_weights = np.cumsum(sorted_weights)
        cutoff = cum_weights[-1] / 2.0
        median_idx = np.searchsorted(cum_weights, cutoff)
        result[i] = sorted_values[median_idx]

    return result


def get_slips_vel_seasonal(time, velocity, drops=True, threshold=0, mindrop=0, threshtype='median', window_size=312,season_radius=2,weight_boost=2,weight_type="flat"):
    """
    Identical to get_slips_wrap & get_slips_core in terms of parameters and outputs, but specifically designed for velocity signals & to ensure no negative sizes.
    Use the trapezoidal rule + chopping off all parts of the signal less than the threshold (i.e. velocity < threshold*std(data) or whatever) to get a more accurate view of the size of an event.

    Parameters
    ----------
    time: (List or array-like; REQUIRED)
        Time vector in data units.
        Defaults to an index array, i.e., an array ranging from 0 to N - 1, where N is the length of the input data.
    velocity: (List or array-like; REQUIRED)
        Time series data to be analyzed for avalanches IF data is some quantity where at each time a new value is acquired.
        I.e., number of spins flipped in one timestep of the random-field Ising model (RFIM) or the number of cell failures in one timestep in a slip model.
        An avalanche in this perspective is a "slip RATE", in the parlance of the Dahmen Group.
    drops: (Boolean; OPTIONAL)
        Default value is TRUE.
        Whether to scan the time series for drops in the data.
    threshold: (Float; OPTIONAL)
        Default value is 0.
        Number of standard deviations above the average velocity a fluctuation must be before it is considered an event.
        Recommend 0 for most applications.
        Setting this equal to -1 forces a zero-velocity threshold on velocity curve.
            This is useful for simulations since there's little to no noise to be mistaken for an avalanche.
    mindrop: (Float; OPTIONAL)
        Default value is 0.
        Minimum size required to be counted as an event.
        Recommend 0 when starting analysis, but should be set higher when i.e. data culling to find the true value of tau.
    threshtype: (String; OPTIONAL)
        Default value is 'median'.
        What type of threshold to use. Options:
        'median'
            -- Uses the median velocity instead of the mean velocity to quantify the average velocity.
            -- Works best in signals with many excursions (i.e., many avalanches) and is not sensitive to outliers.
            -- Threshold is calculated using median absolute deviation (MAD) with this option instead of standard deviation because it more accurately describes the dispersion of the noise fluctuations while ignoring the avalanches.
        'mean'
            -- This is the traditional method.
            -- The threshold is compared to the mean velocity, (displacement[end]-displacement[start])/(time[end]-time[start]).
            -- Threshold is calculated using the standard deviation with this setting.
            -- This method has some major issues in non-simulation environments, as the standard deviation is very sensitive to large excursions from the noise floor
        'sliding_median'
            -- Uses a sliding median of width window_length to obtain a sliding median estimate of the background velocity.
            -- Useful when the background rate of the process is not constant over the course of the experiment.
            -- Threshold is calculated using median absolute deviation with this option and follows the change in average velocity while accurately describing the dispersion of the noise.
    window_size: (Int; OPTIONAL)
        Default value is 1.
        The window size, in datapoints, to be used when calculating the sliding median.
        Should be set to be much longer than the length of an avalanche in your data.
        Jordan found setting window_size = 3% the length of the data (in one case) was good, but this value can be anywhere from just a few hundred datapoints (very short avalanches, many datapoints) to up to 10-20% of the total length of the signal (when the data are shorter but contain several very long avalanches).
        This is worth playing with!

    Returns
    -------
    [0] velocity: list of lists
        -- A list of avalanche velocity curves.
        -- E.g. v[8] will be the velocity curve for the 9th avalanche.
        -- Velocity curve will always begin and end with 0s because the velocity had to cross 0 both times in its run.
    [1] times: list of lists
        -- A list of avalanche time curves.
        -- E.g. t[8] will be the corresponding times at which v[8] velocity curve occured.
        -- The time curve will have the 0s in the velocity curve occur at t_start - ts/2 and t_end + ts/2 as a 0-order estimate of when the curve intersected with zero.
    [2] sizes: list of floats
        -- List of avalanche sizes.
        -- The size is defined as the amount the displacement changes while the velocity is strictly above the threshold.
        -- Each size is corrected by the background rate.
        -- That is, (background rate)*(duration) is removed from the event size.
    [3] durations: list of floats
        -- List of avalanche durations.
        -- The duration is the amount of time the velocity is strictly above the threshold.
        -- For example, an avalanche with velocity profile [0,1,2,3,2,1,0] has a duration of 5 timesteps.
    [4] st: list of ints
        -- List of avalanche start indices.
        -- E.g. st[8] is the index on the displacement where the 9th avalanche occurred.
        -- The start index is the first index that the velocity is above the threshold.
    [5] en: list of ints
        -- List of avalanche end indices.
        -- E.g. en[8] will be the index on the displacement where the 9th avalanche ends.
        -- The end index is the first index after the velocity is below the threshold (accounting for Python index counting).
    """
    trapz = scipy.integrate.trapezoid
    # Like standard deviation but for median
    mad = scipy.stats.median_abs_deviation
    sliding_median = scipy.ndimage.median_filter
    ones = np.ones
    append = np.append
    diff = np.diff
    where = np.where
    arr = np.array
    insert = np.insert
    zeros = np.zeros
    median = np.median

    window_size = window_size + (1 - window_size % 2)

    std = lambda x: 1.4826 * mad(x)
    if threshtype == 'median':
        avg = np.median
        cutoff_velocity = (avg(velocity) + std(velocity) * threshold * (int(threshold != -1))) * ones(len(velocity))
    if threshtype == 'mean':
        avg = np.mean
        std = np.std
        cutoff_velocity = (avg(velocity) + std(velocity) * threshold * (int(threshold != -1))) * ones(len(velocity))
    if threshtype == 'sliding_median':
        #cutoff_velocity = sliding_median(velocity, window_size, mode='nearest')
        cutoff_velocity = seasonal_weighted_median_filter(np.array(velocity), window_size=window_size, season_length=52, season_radius=season_radius, weight_boost=weight_boost, weight_type=weight_type)#Weighted for seasons
        cutoff_velocity[:window_size // 2] = cutoff_velocity[window_size // 2]
        cutoff_velocity[-window_size // 2:] = cutoff_velocity[-window_size // 2]
        cutoff_velocity = cutoff_velocity + std(velocity) * threshold * (1 * threshold != -1)

    # Treat the velocity by removing the trend such that its centered around zero.
    deriv = velocity - cutoff_velocity
    if drops:
        deriv = -velocity

    # Search for rises in the deriv curve.
    # Set all parts of the curve with velocity less than zero to be equal to zero.
    deriv[deriv < 0] = 0
    # Get the slips
    slips = append(0, diff(1 * (deriv > 0)))
    # Velocity start index (first index above 0)
    index_begins = where(slips == 1)[0]
    # Velocity end index (last index above 0)
    index_ends = where(slips == -1)[0]

    if index_begins.size == 0:
        index_begins = arr([0])
    if index_ends.size == 0:
        index_ends = arr([len(time) - 1])
    if index_begins[-1] >= index_ends[-1]:
        index_ends = append(index_ends, len(time) - 1)
    if index_begins[0] >= index_ends[0]:
        index_begins = insert(index_begins, 0, 0)

    # Get the possible sizes
    possible_sizes = zeros(len(index_begins))
    possible_durations = zeros(len(index_begins))
    for i in range(len(index_begins)):
        st = index_begins[i]
        en = index_ends[i]
        trapz_st = max([st - 1, 0])
        trapz_en = min(en + 1, len(deriv))
        possible_sizes[i] = trapz(deriv[trapz_st:trapz_en], time[trapz_st:trapz_en])
        possible_durations[i] = time[en] - time[st]

    idxs = where(possible_sizes > mindrop)[0]
    sizes = possible_sizes[idxs]
    durations = possible_durations[idxs]
    index_av_begins = index_begins[idxs]
    index_av_ends = index_ends[idxs]

    time2 = 0.5 * (time[0:len(time) - 1] + time[1:len(time)])
    # Sampling time
    tsamp = median(diff(time2))
    time2 = append(time2, time2[-1] + tsamp)
    velocity = []
    times = []
    for k in range(len(index_av_begins)):
        st = index_av_begins[k]
        en = index_av_ends[k]
        mask = np.arange(st, en)
        if st == en:
            mask = st

        # First-order approximation: assume the shape begins and ends at min_diff halfway between the start index and the preceeding index.
        curv = zeros(en - st + 2)
        curt = zeros(en - st + 2)
        curv[1:-1] = deriv[mask]
        curt[1:-1] = time2[mask]
        curt[0] = curt[1] - tsamp / 2
        curt[-1] = curt[-2] + tsamp / 2
        velocity.append(list(curv))
        times.append(list(curt))

    return [list(velocity), list(times), list(sizes), list(durations), list(index_av_begins), list(index_av_ends), list(slips), list(deriv)]
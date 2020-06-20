import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

FILTERED_REGIONS = [
    'Virgin Islands',
    'American Samoa',
    'Northern Mariana Islands',
    'Guam',
    'Puerto Rico']


FILE = 'covid19-9141215a245a4dab90a0ff0d2ead2882.csv'

states_names = ('PI','RR', 'AC','MT', 'MS', 'GO', 
	'PR', 'RS', 'SC', 'SP', 'AL', 'AM', 
	'AP', 'BA', 'CE', 'DF', 'ES', 'MA', 
	'MG', 'PA', 'PB', 'PE', 'RJ', 'RN', 
	'RO', 'SE', 'TO')

def main():
	st.title('Estimating COVID-19 spread more precisely')

	# Load data
	raw_states_df = load_data(FILE)

	# Sidebar states
	state = st.sidebar.selectbox("Select a state: ", states_names)

	# Plot examples ####
	original, smoothed = select_cases(raw_states_df, state)
	original.plot(title=f"{state} New Cases per Day",
               c='k',
               linestyle=':',
               alpha=.5,
               label='Actual',
               legend=True,
             figsize=(600/72, 400/72))
	ax = smoothed.plot(label='Smoothed',
	                   legend=True)
	ax.get_figure().set_facecolor('w')
	st.subheader('Example by state: %s' % state)
	st.pyplot()

	# Plot posteriors with following parameters ####

	st.sidebar.title("Customize parameters for posterior simulation")
	k = np.array([20, 40, 55, 90])

	window = st.sidebar.slider("Window", min_value=2, max_value=10, value=7, step=1)
	min_periods = st.sidebar.slider("Min periods", min_value=1, max_value=5, value=1, step=1)
	gamma = st.sidebar.slider("Gamma", min_value=float(0), max_value=float(1), value=float(1/4), step=(.1))
	r_t_max = st.sidebar.slider("$R_t$ max", min_value=1, max_value=20, value=12, step=1)
	r_t_range = np.linspace(0, r_t_max, r_t_max*100+1)

	posteriors = get_posteriors(smoothed, window, min_periods, gamma, r_t_range)

	ax = posteriors.plot(title=f'{state} - Daily Posterior for $R_t$',
						legend=False, 
						lw=1,
						c='k',
						alpha=.3,
						xlim=(0.4,4))
	ax.set_xlabel('$R_t$');

	st.subheader('Plot posterior distribution by state: %s' % state)
	st.pyplot()

# Cache raw data
@st.cache
def load_data(FILE):
	raw_states_df = pd.read_csv(FILE,
		usecols=[1,3,7],
		parse_dates=['date'],
		squeeze=True
		).sort_index()
	raw_states_df['cases']= raw_states_df['last_available_confirmed']
	raw_states_df.drop(['last_available_confirmed'],axis=1, inplace=True)
	raw_states_df
	return raw_states_df

# prepare cases
@st.cache
def prepare_cases(cases):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    zeros = smoothed.index[smoothed.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed.index.get_loc(last_zero) + 1
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed

# Cache smoothed results
@st.cache
def select_cases(raw_states_df, state):
	states_df = raw_states_df.groupby(['state','date']).sum()
	states = states_df.squeeze()
	cases = states.xs(state).rename(f"{state} cases")
	original, smoothed = prepare_cases(cases)
	return original, smoothed

# Cache posteriors distributions
def get_posteriors(sr, window, min_periods, gamma, r_t_range):
    lam = sr[:-1].values * np.exp(gamma * (r_t_range[:, None] - 1))

    # Note: if you want to have a Uniform prior you can use the following line instead.
    # I chose the gamma distribution because of our prior knowledge of the likely value
    # of R_t.
    
    # prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))
    prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)

    likelihoods = pd.DataFrame(
        # Short-hand way of concatenating the prior and likelihoods
        data = np.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],
        index = r_t_range,
        columns = sr.index)

    # Perform a rolling sum of log likelihoods. This is the equivalent
    # of multiplying the original distributions. Exponentiate to move
    # out of log.
    posteriors = likelihoods.rolling(window,
                                     axis=1,
                                     min_periods=min_periods).sum()
    posteriors = np.exp(posteriors)

    # Normalize to 1.0
    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)
    
    return posteriors

if __name__ == "__main__":
	main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import downloaddata

from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from datetime import date

from scipy import stats as sps
from scipy.interpolate import interp1d

# Download most recent data
downloaddata.main()

FILE = "data/Covid-19-Brasil_" + str(date.today()) + ".csv"

states_names = (
	'PE','RR', 'AC','MT', 'MS', 'GO', 
	'PR', 'RS', 'SC', 'SP', 'AL', 'AM', 
	'AP', 'BA', 'CE', 'DF', 'ES', 'MA', 
	'MG', 'PA', 'PB', 'PI', 'RJ', 'RN', 
	'RO', 'SE', 'TO'
)

def main():
	st.title('Estimating COVID-19 spread more precisely')

	# Load data
	raw_states_df = load_data(FILE)

	# Sidebar states
	st.sidebar.title("Select a state:")
	state = st.sidebar.selectbox("", states_names)

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

	# Create sidebar for paramaters
	window = st.sidebar.slider("Window", min_value=2, max_value=10, value=7, step=1)
	min_periods = st.sidebar.slider("Min periods", min_value=1, max_value=5, value=1, step=1)
	gamma = st.sidebar.slider("Gamma", min_value=float(0), max_value=float(1), value=float(1/4), step=(.05))
	r_t_max = st.sidebar.slider("$R_t$ max", min_value=1, max_value=20, value=12, step=1)
	r_t_range = np.linspace(0, r_t_max, r_t_max*100+1)

	# Calculate posteriors
	posteriors = get_posteriors(smoothed, window, min_periods, gamma, r_t_range)

	# Plot posteriors
	ax = posteriors.plot(title=f'{state} - Daily Posterior for $R_t$',
						legend=False, 
						lw=1,
						c='k',
						alpha=.3,
						xlim=(0.4,4))
	ax.set_xlabel('$R_t$');
	st.subheader('Plot posterior distribution by state: %s' % state)
	st.pyplot()

	#
	st.sidebar.title("Calculate high density interval")
	r_t_generate = st.sidebar.checkbox("Run!")
	if r_t_generate:
		hdis = highest_density_interval(posteriors)
		most_likely = posteriors.idxmax().rename('ML')
		# Look into why you shift -1
		result = pd.concat([most_likely, hdis], axis=1)

		st.subheader('Rt Highest Density Interval: %s' % state)

		fig, ax = plt.subplots(figsize=(600/72,400/72))
		plot_rt(result, fig, ax, state)
		st.pyplot()

	# Plot all
	selected_states = st.sidebar.multiselect("Choose other states to compare (this may take a while):",
									states_names, 
									default=None)

	if selected_states:

		#selected_states.append(state)
		df_filtered = raw_states_df[raw_states_df['state'].isin(selected_states)]
		results = compute_results_all_states(df_filtered, window, min_periods, gamma, r_t_range)
		
		if len(selected_states) > 3:
			ncols = 4
		else:
			ncols = 3

		nrows = int(np.ceil(len(results) / ncols))
		# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))
		fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))
		for i, (state_name, result) in enumerate(results.items()):
		    plot_rt(result, fig, axes.flat[i], state_name)
		fig.tight_layout()
		fig.set_facecolor('w')
		st.pyplot()


# Cache raw data
@st.cache(show_spinner=False)
def load_data(FILE):
	raw_states_df = pd.read_csv(FILE,
		usecols=[1,3,7],
		parse_dates=['date'],
		squeeze=True
		).sort_index()
	raw_states_df['cases']= raw_states_df['last_available_confirmed']
	raw_states_df.drop(['last_available_confirmed'],axis=1, inplace=True)
	return raw_states_df

# prepare cases
@st.cache(show_spinner=False)
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
@st.cache(show_spinner=False)
def select_cases(raw_states_df, state):
	states_df = raw_states_df.groupby(['state','date']).sum()
	states = states_df.squeeze()
	cases = states.xs(state).rename(f"{state} cases")
	original, smoothed = prepare_cases(cases)
	return original, smoothed

# Cannot cache posterior distributions - random numbers generated
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

@st.cache(show_spinner=False)
def highest_density_interval(pmf, p=.95):
    low = ''
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col]) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j<best[1]-best[0]):
                best = (i, i+j+1)
                break
    
    # I had to do this because some points in best are None, due to the data quality
    if best is not None:     
        low = pmf.index[best[0]]
        high = pmf.index[best[1]]
    else:
        low = float('nan')
        high = float('nan')

    return pd.Series([low, high], index=['Low', 'High'])

def plot_rt(result, fig, ax, state):

	ax.set_title(f"{state}")

	# Colors
	ABOVE = [1,0,0]
	MIDDLE = [1,1,1]
	BELOW = [0,0,0]
	cmap = ListedColormap(np.r_[
	np.linspace(BELOW,MIDDLE,25),
	np.linspace(MIDDLE,ABOVE,25)
	])
	color_mapped = lambda y: np.clip(y, .5, 1.5)-.5

	index = result['ML'].index.get_level_values('date')
	values = result['ML'].values

	# Plot dots and line
	ax.plot(index, values, c='k', zorder=1, alpha=.25)
	ax.scatter(index,
				values,
				s=40,
				lw=.5,
				c=cmap(color_mapped(values)),
				edgecolors='k', zorder=2)

	# Aesthetically, extrapolate credible interval by 1 day either side
	lowfn = interp1d(date2num(index),
					result['Low'].values,
					bounds_error=False,
					fill_value='extrapolate')

	highfn = interp1d(date2num(index),
					result['High'].values,
					bounds_error=False,
					fill_value='extrapolate')

	extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
	end=index[-1]+pd.Timedelta(days=1))

	ax.fill_between(extended,
					lowfn(date2num(extended)),
					highfn(date2num(extended)),
					color='k',
					alpha=.1,
					lw=0,
					zorder=3)

	ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

	# Formatting
	ax.xaxis.set_major_locator(mdates.MonthLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
	ax.xaxis.set_minor_locator(mdates.DayLocator())
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
	ax.yaxis.tick_right()
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.margins(0)
	ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
	ax.margins(0)
	ax.set_ylim(0.0,3.5)
	ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
	fig.set_facecolor('w')
	ax.set_title(f'Real-time $R_t$ for {state}')
	ax.set_ylim(.5,3.5)
	ax.xaxis.set_major_locator(mdates.WeekdayLocator())
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

def compute_results_all_states(raw_states_df, window, min_periods, gamma, r_t_range):
	FILTERED_REGIONS = []

	results = {}

	states_df = raw_states_df.groupby(['state', 'date']).sum()
	states = states_df.squeeze()

	states_to_process = states.loc[~states.index.get_level_values('state').isin(FILTERED_REGIONS)]

	for state_name, cases in states_to_process.groupby(level='state'):
		print(f'Processing {state_name}')
		new, smoothed = prepare_cases(cases)
		print('\tGetting Posteriors')
		posteriors = get_posteriors(smoothed, window, min_periods, gamma, r_t_range)
		print('\tGetting HDIs')
		hdis = highest_density_interval(posteriors)

		print('\tGetting most likely values')
		most_likely = posteriors.idxmax().rename('ML')
		result = pd.concat([most_likely, hdis], axis=1)
		results[state_name] = result.droplevel(0)

	return results

if __name__ == "__main__":
	main()
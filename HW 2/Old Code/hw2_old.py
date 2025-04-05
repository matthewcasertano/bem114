# BEM 114: Hedge Funds - Problem Set 2
# Building Factor Portfolios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("deep")

# Load the cleaned CRSP data
print("Loading cleaned CRSP data...")
crsp = pd.read_csv('crsp_cleaned.csv')

# Convert date to datetime format
crsp['date'] = pd.to_datetime(crsp['date'])
crsp['year'] = crsp['date'].dt.year
crsp['month'] = crsp['date'].dt.month
crsp['yearmonth'] = crsp['year'] * 100 + crsp['month']

# Load Fama-French factors
print("Loading Fama-French factors...")
ff_factors = pd.read_csv('F-F_Research_Data_Factors.CSV', skiprows=0)
ff_factors.columns = ['yearmonth', 'Mkt-RF', 'SMB', 'HML', 'RF']
# Convert percentages to decimals
for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
    ff_factors[col] = ff_factors[col].astype(str).str.strip()
    ff_factors[col] = pd.to_numeric(ff_factors[col], errors='coerce') / 100.0

# Load Fama-French 5 factors (for later questions)
try:
    ff5_factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3.CSV', skiprows=0)
    ff5_factors.columns = ['yearmonth', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        ff5_factors[col] = ff5_factors[col].astype(str).str.strip()
        ff5_factors[col] = pd.to_numeric(ff5_factors[col], errors='coerce') / 100.0
    print("Loaded FF5 factors")
except:
    print("FF5 factors not found, will estimate models without them")
    
# Load Momentum factor
try:
    mom_factor = pd.read_csv('F-F_Momentum_Factor.CSV', skiprows=0)
    mom_factor.columns = ['yearmonth', 'Mom']
    mom_factor['Mom'] = mom_factor['Mom'].astype(str).str.strip()
    mom_factor['Mom'] = pd.to_numeric(mom_factor['Mom'], errors='coerce') / 100.0
    print("Loaded Momentum factor")
except:
    print("Momentum factor not found, will create it ourselves")

# Convert RET to float
crsp['RET'] = pd.to_numeric(crsp['RET'], errors='coerce')

# Calculate market cap (SIZE) for each stock-month
crsp['SIZE'] = abs(pd.to_numeric(crsp['PRC'], errors='coerce')) * pd.to_numeric(crsp['SHROUT'], errors='coerce') / 1000

# Function to calculate performance metrics
def calc_performance(returns):
    """Calculate mean, volatility, and Sharpe ratio for a return series"""
    mean_ret = returns.mean() * 12  # Annualized
    vol = returns.std() * np.sqrt(12)  # Annualized
    sharpe = mean_ret / vol if vol != 0 else 0
    return pd.Series({'Mean (annual)': mean_ret, 'Volatility (annual)': vol, 'Sharpe Ratio': sharpe})

# Function to estimate factor models
def estimate_factor_models(portfolio_returns, factors_df, model_name='CAPM'):
    """Estimate factor models and return alpha, t-stat, and R-squared"""
    if model_name == 'CAPM':
        X = sm.add_constant(factors_df[['Mkt-RF']])
        y = portfolio_returns - factors_df['RF']
    elif model_name == 'FF3':
        X = sm.add_constant(factors_df[['Mkt-RF', 'SMB', 'HML']])
        y = portfolio_returns - factors_df['RF']
    elif model_name == 'FF5':
        X = sm.add_constant(factors_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
        y = portfolio_returns - factors_df['RF']
    elif model_name == 'FF5+Mom':
        X = sm.add_constant(factors_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']])
        y = portfolio_returns - factors_df['RF']
    
    model = sm.OLS(y, X).fit()
    alpha = model.params['const'] * 12  # Annualized alpha
    t_stat = model.tvalues['const']
    r_squared = model.rsquared
    
    return pd.Series({
        f'Alpha ({model_name})': alpha,
        f't-stat ({model_name})': t_stat,
        f'R-squared ({model_name})': r_squared
    })

# ============================================================================
# Question 1: Data Cleaning and Summary Statistics
# ============================================================================
print("\n--- Question 1: Data Cleaning and Summary Statistics ---")

# 1b. Plot the number of listed firms per month
print("Plotting number of listed firms per month...")

# Count unique PERMNOs per month
monthly_firms = crsp.groupby('yearmonth')['PERMNO'].nunique().reset_index()
monthly_firms['date'] = pd.to_datetime(monthly_firms['yearmonth'].astype(str), format='%Y%m')

plt.figure(figsize=(12, 6))
plt.plot(monthly_firms['date'], monthly_firms['PERMNO'])
plt.title('Number of Listed Firms per Month (1926-2020)')
plt.xlabel('Year')
plt.ylabel('Number of Firms')
plt.grid(True)
plt.tight_layout()
plt.savefig('listed_firms_over_time.png')
print("Plot saved as 'listed_firms_over_time.png'")

# ============================================================================
# Question 2: Replicate Size Factor
# ============================================================================
print("\n--- Question 2: Replicate Size Factor ---")

# Function to form size portfolios
def form_size_portfolios(data, n_portfolios=10):
    """Form portfolios based on market capitalization (SIZE)"""
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Group by date
    grouped = df.groupby('yearmonth')
    
    # Initialize lists to store results
    portfolio_assignments = []
    
    # For each month, assign stocks to portfolios based on SIZE
    for name, group in grouped:
        # Remove stocks with missing SIZE or RET
        valid_stocks = group.dropna(subset=['SIZE', 'RET'])
        
        if len(valid_stocks) == 0:
            continue
        
        # Calculate SIZE percentiles for this month
        for i in range(n_portfolios):
            lower_percentile = i / n_portfolios * 100
            upper_percentile = (i + 1) / n_portfolios * 100
            
            # Find stocks in this percentile range
            if i == 0:  # First portfolio includes the lower bound
                mask = (valid_stocks['SIZE'] <= np.percentile(valid_stocks['SIZE'], upper_percentile))
            elif i == n_portfolios - 1:  # Last portfolio includes the upper bound
                mask = (valid_stocks['SIZE'] > np.percentile(valid_stocks['SIZE'], lower_percentile))
            else:  # Middle portfolios
                mask = ((valid_stocks['SIZE'] > np.percentile(valid_stocks['SIZE'], lower_percentile)) & 
                         (valid_stocks['SIZE'] <= np.percentile(valid_stocks['SIZE'], upper_percentile)))
            
            # Assign portfolio number
            portfolio_stocks = valid_stocks[mask].copy()
            portfolio_stocks['portfolio'] = i + 1  # Portfolio numbers from 1 to n_portfolios
            
            portfolio_assignments.append(portfolio_stocks)
    
    # Combine all assignments
    all_assignments = pd.concat(portfolio_assignments)
    
    return all_assignments

# Form size portfolios
print("Forming size portfolios...")
size_portfolios = form_size_portfolios(crsp)

# Calculate equal-weighted portfolio returns
print("Calculating equal-weighted portfolio returns...")
equal_weighted_size = size_portfolios.groupby(['yearmonth', 'portfolio'])['RET'].mean().reset_index()
equal_weighted_size_pivot = equal_weighted_size.pivot(index='yearmonth', columns='portfolio', values='RET')
equal_weighted_size_pivot.columns = [f'EW_P{i}' for i in equal_weighted_size_pivot.columns]

# Calculate value-weighted portfolio returns
print("Calculating value-weighted portfolio returns...")
# First, calculate weighted returns
size_portfolios['weighted_ret'] = size_portfolios['RET'] * size_portfolios['SIZE']
# Group by yearmonth and portfolio, and sum weights and weighted returns
value_weighted_sum = size_portfolios.groupby(['yearmonth', 'portfolio'])[['weighted_ret', 'SIZE']].sum()
# Calculate value-weighted returns
value_weighted_size = (value_weighted_sum['weighted_ret'] / value_weighted_sum['SIZE']).reset_index()
value_weighted_size_pivot = value_weighted_size.pivot(index='yearmonth', columns='portfolio', values=0)
value_weighted_size_pivot.columns = [f'VW_P{i}' for i in value_weighted_size_pivot.columns]

# Combine all portfolio returns
size_returns = pd.concat([equal_weighted_size_pivot, value_weighted_size_pivot], axis=1)

# Calculate mean returns for each decile
mean_size_returns_ew = equal_weighted_size_pivot.mean() * 12  # Annualized
mean_size_returns_vw = value_weighted_size_pivot.mean() * 12  # Annualized

# Form long-short SMB portfolio
size_returns['EW_SMB'] = size_returns['EW_P1'] - size_returns['EW_P10']  # Small minus Big (equal-weighted)
size_returns['VW_SMB'] = size_returns['VW_P1'] - size_returns['VW_P10']  # Small minus Big (value-weighted)

# Calculate performance metrics
size_performance_ew = calc_performance(size_returns['EW_SMB'])
size_performance_vw = calc_performance(size_returns['VW_SMB'])

# Estimate CAPM and FF3 models
# First merge with factors
size_returns_with_factors = pd.merge(size_returns, ff_factors, left_index=True, right_on='yearmonth', how='inner')

# Estimate models
size_capm_ew = estimate_factor_models(size_returns_with_factors['EW_SMB'], size_returns_with_factors, 'CAPM')
size_ff3_ew = estimate_factor_models(size_returns_with_factors['EW_SMB'], size_returns_with_factors, 'FF3')
size_capm_vw = estimate_factor_models(size_returns_with_factors['VW_SMB'], size_returns_with_factors, 'CAPM')
size_ff3_vw = estimate_factor_models(size_returns_with_factors['VW_SMB'], size_returns_with_factors, 'FF3')

# Analyze post-publication performance (after 1992)
post_publication = size_returns_with_factors[size_returns_with_factors['yearmonth'] >= 199207]
size_post_pub_ew = calc_performance(post_publication['EW_SMB'])
size_post_pub_vw = calc_performance(post_publication['VW_SMB'])

# Analyze post-Dot-Com performance (after 2002)
post_dotcom = size_returns_with_factors[size_returns_with_factors['yearmonth'] >= 200201]
size_post_dotcom_ew = calc_performance(post_dotcom['EW_SMB'])
size_post_dotcom_vw = calc_performance(post_dotcom['VW_SMB'])

# ============================================================================
# Question 3: Replicate Momentum
# ============================================================================
print("\n--- Question 3: Replicate Momentum Factor ---")

# Function to form momentum portfolios
def form_momentum_portfolios(data, n_portfolios=10, lookback_period=11, skip_month=1):
    """Form portfolios based on past returns (momentum)"""
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Create a unique identifier for sorting
    df['sort_id'] = df['PERMNO'].astype(str) + "_" + df['yearmonth'].astype(str)
    
    # Group by PERMNO to calculate rolling returns
    print("Calculating rolling cumulative returns...")
    grouped = df.sort_values(['PERMNO', 'yearmonth']).groupby('PERMNO')
    
    # List to store results
    results = []
    
    # Process each stock group
    for name, group in grouped:
        # Ensure the group is sorted by date
        group = group.sort_values('yearmonth')
        
        # Calculate rolling returns over specified lookback period
        # We use a minimum of lookback_period/2 observations to avoid too many NAs
        group['past_return'] = group['RET'].rolling(window=lookback_period, min_periods=int(lookback_period/2)).apply(
            lambda x: np.prod(1 + x) - 1, raw=True
        )
        
        # Shift by skip_month to implement the skip-month strategy
        group['past_return'] = group['past_return'].shift(skip_month)
        
        results.append(group)
    
    # Combine all stocks
    momentum_data = pd.concat(results)
    
    # Group by date
    grouped = momentum_data.groupby('yearmonth')
    
    # Initialize lists to store portfolio assignments
    portfolio_assignments = []
    
    # For each month, assign stocks to portfolios based on past returns
    for name, group in grouped:
        # Remove stocks with missing past_return or RET
        valid_stocks = group.dropna(subset=['past_return', 'RET'])
        
        if len(valid_stocks) < n_portfolios:
            continue
        
        # Calculate past_return percentiles for this month
        for i in range(n_portfolios):
            lower_percentile = i / n_portfolios * 100
            upper_percentile = (i + 1) / n_portfolios * 100
            
            # Find stocks in this percentile range
            if i == 0:  # First portfolio includes the lower bound
                mask = (valid_stocks['past_return'] <= np.percentile(valid_stocks['past_return'], upper_percentile))
            elif i == n_portfolios - 1:  # Last portfolio includes the upper bound
                mask = (valid_stocks['past_return'] > np.percentile(valid_stocks['past_return'], lower_percentile))
            else:  # Middle portfolios
                mask = ((valid_stocks['past_return'] > np.percentile(valid_stocks['past_return'], lower_percentile)) & 
                         (valid_stocks['past_return'] <= np.percentile(valid_stocks['past_return'], upper_percentile)))
            
            # Assign portfolio number
            portfolio_stocks = valid_stocks[mask].copy()
            portfolio_stocks['portfolio'] = i + 1  # Portfolio numbers from 1 to n_portfolios
            
            portfolio_assignments.append(portfolio_stocks)
    
    # Combine all assignments
    all_assignments = pd.concat(portfolio_assignments)
    
    return all_assignments

# Form momentum portfolios
print("Forming momentum portfolios...")
momentum_portfolios = form_momentum_portfolios(crsp)

# Calculate equal-weighted portfolio returns
print("Calculating equal-weighted momentum portfolio returns...")
equal_weighted_mom = momentum_portfolios.groupby(['yearmonth', 'portfolio'])['RET'].mean().reset_index()
equal_weighted_mom_pivot = equal_weighted_mom.pivot(index='yearmonth', columns='portfolio', values='RET')
equal_weighted_mom_pivot.columns = [f'EW_P{i}' for i in equal_weighted_mom_pivot.columns]

# Calculate value-weighted portfolio returns
print("Calculating value-weighted momentum portfolio returns...")
# First, calculate weighted returns
momentum_portfolios['weighted_ret'] = momentum_portfolios['RET'] * momentum_portfolios['SIZE']
# Group by yearmonth and portfolio, and sum weights and weighted returns
value_weighted_sum = momentum_portfolios.groupby(['yearmonth', 'portfolio'])[['weighted_ret', 'SIZE']].sum()
# Calculate value-weighted returns
value_weighted_mom = (value_weighted_sum['weighted_ret'] / value_weighted_sum['SIZE']).reset_index()
value_weighted_mom_pivot = value_weighted_mom.pivot(index='yearmonth', columns='portfolio', values=0)
value_weighted_mom_pivot.columns = [f'VW_P{i}' for i in value_weighted_mom_pivot.columns]

# Combine all portfolio returns
mom_returns = pd.concat([equal_weighted_mom_pivot, value_weighted_mom_pivot], axis=1)

# Calculate mean returns for each decile
mean_mom_returns_ew = equal_weighted_mom_pivot.mean() * 12  # Annualized
mean_mom_returns_vw = value_weighted_mom_pivot.mean() * 12  # Annualized

# Form long-short WML portfolio
mom_returns['EW_WML'] = mom_returns['EW_P10'] - mom_returns['EW_P1']  # Winners minus Losers (equal-weighted)
mom_returns['VW_WML'] = mom_returns['VW_P10'] - mom_returns['VW_P1']  # Winners minus Losers (value-weighted)

# Calculate performance metrics
mom_performance_ew = calc_performance(mom_returns['EW_WML'])
mom_performance_vw = calc_performance(mom_returns['VW_WML'])

# Estimate CAPM, FF3, and FF5 models
# First merge with factors
mom_returns_with_factors = pd.merge(mom_returns, ff_factors, left_index=True, right_on='yearmonth', how='inner')

# Create a WML factor for FF5+Mom model
if 'ff5_factors' in locals() and 'mom_factor' in locals():
    ff5_mom_factors = pd.merge(ff5_factors, mom_factor, on='yearmonth', how='inner')
    mom_returns_with_all_factors = pd.merge(mom_returns, ff5_mom_factors, left_index=True, right_on='yearmonth', how='inner')
    
    # Estimate models
    mom_capm_ew = estimate_factor_models(mom_returns_with_factors['EW_WML'], mom_returns_with_factors, 'CAPM')
    mom_ff3_ew = estimate_factor_models(mom_returns_with_factors['EW_WML'], mom_returns_with_factors, 'FF3')
    mom_ff5_ew = estimate_factor_models(mom_returns_with_all_factors['EW_WML'], mom_returns_with_all_factors, 'FF5')
    
    mom_capm_vw = estimate_factor_models(mom_returns_with_factors['VW_WML'], mom_returns_with_factors, 'CAPM')
    mom_ff3_vw = estimate_factor_models(mom_returns_with_factors['VW_WML'], mom_returns_with_factors, 'FF3')
    mom_ff5_vw = estimate_factor_models(mom_returns_with_all_factors['VW_WML'], mom_returns_with_all_factors, 'FF5')
else:
    # Just estimate CAPM and FF3 if FF5 factors unavailable
    mom_capm_ew = estimate_factor_models(mom_returns_with_factors['EW_WML'], mom_returns_with_factors, 'CAPM')
    mom_ff3_ew = estimate_factor_models(mom_returns_with_factors['EW_WML'], mom_returns_with_factors, 'FF3')
    
    mom_capm_vw = estimate_factor_models(mom_returns_with_factors['VW_WML'], mom_returns_with_factors, 'CAPM')
    mom_ff3_vw = estimate_factor_models(mom_returns_with_factors['VW_WML'], mom_returns_with_factors, 'FF3')

# ============================================================================
# Question 4: Replicate Betting-Against-Beta
# ============================================================================
print("\n--- Question 4: Replicate Betting-Against-Beta ---")

# Function to form BAB portfolios
def form_bab_portfolios(data, market_factor, n_portfolios=10, estimation_window=36):
    """Form portfolios based on estimated CAPM betas"""
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Merge with market factor
    market = market_factor[['yearmonth', 'Mkt-RF', 'RF']].copy()
    
    # Initialize list to store all stocks with beta estimates
    all_stock_betas = []
    
    # Process each stock
    print("Estimating rolling betas...")
    for permno in df['PERMNO'].unique():
        # Get data for this stock
        stock_data = df[df['PERMNO'] == permno].sort_values('yearmonth')
        
        # Skip if not enough data
        if len(stock_data) < estimation_window:
            continue
            
        # Merge with market returns
        stock_with_mkt = pd.merge(stock_data, market, on='yearmonth', how='inner')
        
        # Skip if not enough data after merge
        if len(stock_with_mkt) < estimation_window:
            continue
            
        # Calculate excess returns
        stock_with_mkt['ExRet'] = stock_with_mkt['RET'] - stock_with_mkt['RF']
        
        # For each month, calculate beta using previous 36 months
        for i in range(estimation_window, len(stock_with_mkt)):
            window = stock_with_mkt.iloc[i-estimation_window:i]
            
            try:
                # Run regression to get beta
                X = sm.add_constant(window['Mkt-RF'])
                y = window['ExRet']
                model = sm.OLS(y, X).fit()
                beta = model.params['Mkt-RF']
                
                # Current month data
                current_month = stock_with_mkt.iloc[i].copy()
                current_month['beta'] = beta
                
                all_stock_betas.append(current_month)
            except:
                # Skip if regression fails (e.g., not enough non-NA values)
                continue
    
    # Combine all stocks with beta estimates
    beta_data = pd.DataFrame(all_stock_betas)
    
    # Group by date for portfolio formation
    grouped = beta_data.groupby('yearmonth')
    
    # Initialize lists to store portfolio assignments
    portfolio_assignments = []
    
    # For each month, assign stocks to portfolios based on betas
    for name, group in grouped:
        # Remove stocks with missing beta or RET
        valid_stocks = group.dropna(subset=['beta', 'RET'])
        
        if len(valid_stocks) < n_portfolios:
            continue
        
        # Calculate beta percentiles for this month
        for i in range(n_portfolios):
            lower_percentile = i / n_portfolios * 100
            upper_percentile = (i + 1) / n_portfolios * 100
            
            # Find stocks in this percentile range
            if i == 0:  # First portfolio includes the lower bound
                mask = (valid_stocks['beta'] <= np.percentile(valid_stocks['beta'], upper_percentile))
            elif i == n_portfolios - 1:  # Last portfolio includes the upper bound
                mask = (valid_stocks['beta'] > np.percentile(valid_stocks['beta'], lower_percentile))
            else:  # Middle portfolios
                mask = ((valid_stocks['beta'] > np.percentile(valid_stocks['beta'], lower_percentile)) & 
                         (valid_stocks['beta'] <= np.percentile(valid_stocks['beta'], upper_percentile)))
            
            # Assign portfolio number
            portfolio_stocks = valid_stocks[mask].copy()
            portfolio_stocks['portfolio'] = i + 1  # Portfolio numbers from 1 to n_portfolios
            
            portfolio_assignments.append(portfolio_stocks)
    
    # Combine all assignments
    all_assignments = pd.concat(portfolio_assignments)
    
    return all_assignments
    
    # Group by date
    grouped = beta_data.groupby('yearmonth')
    
    # Initialize lists to store portfolio assignments
    portfolio_assignments = []
    
    # For each month, assign stocks to portfolios based on betas
    for name, group in grouped:
        # Remove stocks with missing beta or RET
        valid_stocks = group.dropna(subset=['beta', 'RET'])
        
        if len(valid_stocks) < n_portfolios:
            continue
        
        # Calculate beta percentiles for this month
        for i in range(n_portfolios):
            lower_percentile = i / n_portfolios * 100
            upper_percentile = (i + 1) / n_portfolios * 100
            
            # Find stocks in this percentile range
            if i == 0:  # First portfolio includes the lower bound
                mask = (valid_stocks['beta'] <= np.percentile(valid_stocks['beta'], upper_percentile))
            elif i == n_portfolios - 1:  # Last portfolio includes the upper bound
                mask = (valid_stocks['beta'] > np.percentile(valid_stocks['beta'], lower_percentile))
            else:  # Middle portfolios
                mask = ((valid_stocks['beta'] > np.percentile(valid_stocks['beta'], lower_percentile)) & 
                         (valid_stocks['beta'] <= np.percentile(valid_stocks['beta'], upper_percentile)))
            
            # Assign portfolio number
            portfolio_stocks = valid_stocks[mask].copy()
            portfolio_stocks['portfolio'] = i + 1  # Portfolio numbers from 1 to n_portfolios
            
            portfolio_assignments.append(portfolio_stocks)
    
    # Combine all assignments
    all_assignments = pd.concat(portfolio_assignments)
    
    return all_assignments

# Form BAB portfolios
print("Forming BAB portfolios...")
bab_portfolios = form_bab_portfolios(crsp, ff_factors)

# Calculate equal-weighted portfolio returns
print("Calculating equal-weighted BAB portfolio returns...")
equal_weighted_bab = bab_portfolios.groupby(['yearmonth', 'portfolio'])['RET'].mean().reset_index()
equal_weighted_bab_pivot = equal_weighted_bab.pivot(index='yearmonth', columns='portfolio', values='RET')
equal_weighted_bab_pivot.columns = [f'EW_P{i}' for i in equal_weighted_bab_pivot.columns]

# Calculate value-weighted portfolio returns
print("Calculating value-weighted BAB portfolio returns...")
# First, calculate weighted returns
bab_portfolios['weighted_ret'] = bab_portfolios['RET'] * bab_portfolios['SIZE']
# Group by yearmonth and portfolio, and sum weights and weighted returns
value_weighted_sum = bab_portfolios.groupby(['yearmonth', 'portfolio'])[['weighted_ret', 'SIZE']].sum()
# Calculate value-weighted returns
value_weighted_bab = (value_weighted_sum['weighted_ret'] / value_weighted_sum['SIZE']).reset_index()
value_weighted_bab_pivot = value_weighted_bab.pivot(index='yearmonth', columns='portfolio', values=0)
value_weighted_bab_pivot.columns = [f'VW_P{i}' for i in value_weighted_bab_pivot.columns]

# Combine all portfolio returns
bab_returns = pd.concat([equal_weighted_bab_pivot, value_weighted_bab_pivot], axis=1)

# Calculate mean returns for each decile
mean_bab_returns_ew = equal_weighted_bab_pivot.mean() * 12  # Annualized
mean_bab_returns_vw = value_weighted_bab_pivot.mean() * 12  # Annualized

# Form long-short BAB portfolio
bab_returns['EW_BAB'] = bab_returns['EW_P1'] - bab_returns['EW_P10']  # Low minus High beta (equal-weighted)
bab_returns['VW_BAB'] = bab_returns['VW_P1'] - bab_returns['VW_P10']  # Low minus High beta (value-weighted)

# Calculate performance metrics
bab_performance_ew = calc_performance(bab_returns['EW_BAB'])
bab_performance_vw = calc_performance(bab_returns['VW_BAB'])

# Estimate CAPM, FF3, FF5, and FF5+Mom models
# First merge with factors
bab_returns_with_factors = pd.merge(bab_returns, ff_factors, left_index=True, right_on='yearmonth', how='inner')

# If we have FF5 and Mom factors, estimate all models
if 'ff5_factors' in locals() and 'mom_factor' in locals():
    ff5_mom_factors = pd.merge(ff5_factors, mom_factor, on='yearmonth', how='inner')
    bab_returns_with_all_factors = pd.merge(bab_returns, ff5_mom_factors, left_index=True, right_on='yearmonth', how='inner')
    
    # Estimate models
    bab_capm_ew = estimate_factor_models(bab_returns_with_factors['EW_BAB'], bab_returns_with_factors, 'CAPM')
    bab_ff3_ew = estimate_factor_models(bab_returns_with_factors['EW_BAB'], bab_returns_with_factors, 'FF3')
    bab_ff5_ew = estimate_factor_models(bab_returns_with_all_factors['EW_BAB'], bab_returns_with_all_factors, 'FF5')
    bab_ff5mom_ew = estimate_factor_models(bab_returns_with_all_factors['EW_BAB'], bab_returns_with_all_factors, 'FF5+Mom')
    
    bab_capm_vw = estimate_factor_models(bab_returns_with_factors['VW_BAB'], bab_returns_with_factors, 'CAPM')
    bab_ff3_vw = estimate_factor_models(bab_returns_with_factors['VW_BAB'], bab_returns_with_factors, 'FF3')
    bab_ff5_vw = estimate_factor_models(bab_returns_with_all_factors['VW_BAB'], bab_returns_with_all_factors, 'FF5')
    bab_ff5mom_vw = estimate_factor_models(bab_returns_with_all_factors['VW_BAB'], bab_returns_with_all_factors, 'FF5+Mom')
else:
    # Just estimate CAPM and FF3 if FF5 factors unavailable
    bab_capm_ew = estimate_factor_models(bab_returns_with_factors['EW_BAB'], bab_returns_with_factors, 'CAPM')
    bab_ff3_ew = estimate_factor_models(bab_returns_with_factors['EW_BAB'], bab_returns_with_factors, 'FF3')
    
    bab_capm_vw = estimate_factor_models(bab_returns_with_factors['VW_BAB'], bab_returns_with_factors, 'CAPM')
    bab_ff3_vw = estimate_factor_models(bab_returns_with_factors['VW_BAB'], bab_returns_with_factors, 'FF3')

# ============================================================================
# Output Results
# ============================================================================
print("\n--- Saving Results ---")

# Create a results directory if it doesn't exist
import os
if not os.path.exists('results'):
    os.makedirs('results')

# Output results for Question 1
# (The plot was already saved)

# Output results for Question 2
# 2b. Mean monthly returns for each size decile
plt.figure(figsize=(10, 6))
plt.plot(mean_size_returns_ew.index, mean_size_returns_ew.values, 'o-', label='Equal-Weighted')
plt.plot(mean_size_returns_vw.index, mean_size_returns_vw.values, 's-', label='Value-Weighted')
plt.title('Mean Annual Returns by Size Decile')
plt.xlabel('Portfolio (1=Small, 10=Big)')
plt.ylabel('Annual Return')
plt.legend()
plt.grid(True)
plt.savefig('results/size_decile_returns.png')

# 2c. Performance metrics for SMB portfolio
size_performance = pd.concat([size_performance_ew, size_performance_vw], axis=1)
size_performance.columns = ['Equal-Weighted SMB', 'Value-Weighted SMB']
size_performance.to_csv('results/size_performance.csv')

# 2d. Factor model results
size_factor_models_ew = pd.concat([size_capm_ew, size_ff3_ew])
size_factor_models_vw = pd.concat([size_capm_vw, size_ff3_vw])
size_factor_models = pd.concat([size_factor_models_ew, size_factor_models_vw], axis=1)
size_factor_models.columns = ['Equal-Weighted SMB', 'Value-Weighted SMB']
size_factor_models.to_csv('results/size_factor_models.csv')

# 2e. Post-publication performance
size_post_publication = pd.DataFrame({
    'Full Sample': pd.concat([size_performance_ew, size_performance_vw]),
    'Post-Publication (1992+)': pd.concat([size_post_pub_ew, size_post_pub_vw]),
    'Post-Dot-Com (2002+)': pd.concat([size_post_dotcom_ew, size_post_dotcom_vw])
})
size_post_publication.to_csv('results/size_post_publication.csv')

# Time series plot of SMB returns
plt.figure(figsize=(12, 6))
plt.plot(size_returns_with_factors['yearmonth'], size_returns_with_factors['EW_SMB'].rolling(window=12).mean(), label='Equal-Weighted SMB (12m MA)')
plt.plot(size_returns_with_factors['yearmonth'], size_returns_with_factors['VW_SMB'].rolling(window=12).mean(), label='Value-Weighted SMB (12m MA)')
plt.axvline(x=199207, color='r', linestyle='--', label='Fama-French 1992 Publication')
plt.axvline(x=200201, color='g', linestyle='--', label='Post Dot-Com')
plt.title('Size Factor Returns Over Time (12-month Moving Average)')
plt.xlabel('Year')
plt.ylabel('Monthly Return')
plt.legend()
plt.grid(True)
plt.savefig('results/size_returns_over_time.png')

# Output results for Question 3
# 3b. Mean monthly returns for each momentum decile
plt.figure(figsize=(10, 6))
plt.plot(mean_mom_returns_ew.index, mean_mom_returns_ew.values, 'o-', label='Equal-Weighted')
plt.plot(mean_mom_returns_vw.index, mean_mom_returns_vw.values, 's-', label='Value-Weighted')
plt.title('Mean Annual Returns by Momentum Decile')
plt.xlabel('Portfolio (1=Losers, 10=Winners)')
plt.ylabel('Annual Return')
plt.legend()
plt.grid(True)
plt.savefig('results/momentum_decile_returns.png')

# 3c. Performance metrics for WML portfolio
mom_performance = pd.concat([mom_performance_ew, mom_performance_vw], axis=1)
mom_performance.columns = ['Equal-Weighted WML', 'Value-Weighted WML']
mom_performance.to_csv('results/momentum_performance.csv')

# 3d. Factor model results
if 'mom_ff5_ew' in locals():
    mom_factor_models_ew = pd.concat([mom_capm_ew, mom_ff3_ew, mom_ff5_ew])
    mom_factor_models_vw = pd.concat([mom_capm_vw, mom_ff3_vw, mom_ff5_vw])
else:
    mom_factor_models_ew = pd.concat([mom_capm_ew, mom_ff3_ew])
    mom_factor_models_vw = pd.concat([mom_capm_vw, mom_ff3_vw])
    
mom_factor_models = pd.concat([mom_factor_models_ew, mom_factor_models_vw], axis=1)
mom_factor_models.columns = ['Equal-Weighted WML', 'Value-Weighted WML']
mom_factor_models.to_csv('results/momentum_factor_models.csv')

# Output results for Question 4
# 4b. Mean monthly returns for each beta decile
plt.figure(figsize=(10, 6))
plt.plot(mean_bab_returns_ew.index, mean_bab_returns_ew.values, 'o-', label='Equal-Weighted')
plt.plot(mean_bab_returns_vw.index, mean_bab_returns_vw.values, 's-', label='Value-Weighted')
plt.title('Mean Annual Returns by Beta Decile')
plt.xlabel('Portfolio (1=Low Beta, 10=High Beta)')
plt.ylabel('Annual Return')
plt.legend()
plt.grid(True)
plt.savefig('results/bab_decile_returns.png')

# 4c. Performance metrics for BAB portfolio
bab_performance = pd.concat([bab_performance_ew, bab_performance_vw], axis=1)
bab_performance.columns = ['Equal-Weighted BAB', 'Value-Weighted BAB']
bab_performance.to_csv('results/bab_performance.csv')

# 4d. Factor model results
if 'bab_ff5_ew' in locals():
    bab_factor_models_ew = pd.concat([bab_capm_ew, bab_ff3_ew, bab_ff5_ew, bab_ff5mom_ew])
    bab_factor_models_vw = pd.concat([bab_capm_vw, bab_ff3_vw, bab_ff5_vw, bab_ff5mom_vw])
else:
    bab_factor_models_ew = pd.concat([bab_capm_ew, bab_ff3_ew])
    bab_factor_models_vw = pd.concat([bab_capm_vw, bab_ff3_vw])
    
bab_factor_models = pd.concat([bab_factor_models_ew, bab_factor_models_vw], axis=1)
bab_factor_models.columns = ['Equal-Weighted BAB', 'Value-Weighted BAB']
bab_factor_models.to_csv('results/bab_factor_models.csv')

# Generate summary report
with open('results/summary_report.md', 'w') as f:
    f.write('# BEM 114: Hedge Funds - Problem Set 2 Summary Report\n\n')
    
    # Question 1
    f.write('## Question 1: Data Cleaning and Summary Statistics\n\n')
    f.write('The data has been cleaned according to the required criteria, and the number of listed firms over time has been plotted in `listed_firms_over_time.png`.\n\n')
    
    # Question 2
    f.write('## Question 2: Size Factor\n\n')
    
    f.write('### Mean Returns by Size Decile\n')
    f.write('| Portfolio | Equal-Weighted | Value-Weighted |\n')
    f.write('|-----------|----------------|----------------|\n')
    for i in range(1, 11):
        f.write(f'| {i} | {mean_size_returns_ew[f"EW_P{i}"]:.4f} | {mean_size_returns_vw[f"VW_P{i}"]:.4f} |\n')
    f.write('\n')
    
    f.write('### SMB Portfolio Performance\n')
    f.write('| Metric | Equal-Weighted | Value-Weighted |\n')
    f.write('|--------|----------------|----------------|\n')
    for idx in size_performance.index:
        f.write(f'| {idx} | {size_performance["Equal-Weighted SMB"][idx]:.4f} | {size_performance["Value-Weighted SMB"][idx]:.4f} |\n')
    f.write('\n')
    
    f.write('### Factor Model Alphas\n')
    f.write('| Model | Equal-Weighted | Value-Weighted |\n')
    f.write('|-------|----------------|----------------|\n')
    for idx in size_factor_models.index:
        if 'Alpha' in idx:
            f.write(f'| {idx} | {size_factor_models["Equal-Weighted SMB"][idx]:.4f} | {size_factor_models["Value-Weighted SMB"][idx]:.4f} |\n')
    f.write('\n')
    
    f.write('### Post-Publication Performance\n')
    f.write('| Period | Equal-Weighted Mean | Value-Weighted Mean |\n')
    f.write('|--------|---------------------|---------------------|\n')
    f.write(f'| Full Sample | {size_performance["Equal-Weighted SMB"]["Mean (annual)"]:.4f} | {size_performance["Value-Weighted SMB"]["Mean (annual)"]:.4f} |\n')
    f.write(f'| Post-Publication (1992+) | {size_post_pub_ew["Mean (annual)"]:.4f} | {size_post_pub_vw["Mean (annual)"]:.4f} |\n')
    f.write(f'| Post-Dot-Com (2002+) | {size_post_dotcom_ew["Mean (annual)"]:.4f} | {size_post_dotcom_vw["Mean (annual)"]:.4f} |\n')
    f.write('\n')
    
    # Question 3
    f.write('## Question 3: Momentum Factor\n\n')
    
    f.write('### Mean Returns by Momentum Decile\n')
    f.write('| Portfolio | Equal-Weighted | Value-Weighted |\n')
    f.write('|-----------|----------------|----------------|\n')
    for i in range(1, 11):
        f.write(f'| {i} | {mean_mom_returns_ew[f"EW_P{i}"]:.4f} | {mean_mom_returns_vw[f"VW_P{i}"]:.4f} |\n')
    f.write('\n')
    
    f.write('### WML Portfolio Performance\n')
    f.write('| Metric | Equal-Weighted | Value-Weighted |\n')
    f.write('|--------|----------------|----------------|\n')
    for idx in mom_performance.index:
        f.write(f'| {idx} | {mom_performance["Equal-Weighted WML"][idx]:.4f} | {mom_performance["Value-Weighted WML"][idx]:.4f} |\n')
    f.write('\n')
    
    f.write('### Factor Model Alphas\n')
    f.write('| Model | Equal-Weighted | Value-Weighted |\n')
    f.write('|-------|----------------|----------------|\n')
    for idx in mom_factor_models.index:
        if 'Alpha' in idx:
            f.write(f'| {idx} | {mom_factor_models["Equal-Weighted WML"][idx]:.4f} | {mom_factor_models["Value-Weighted WML"][idx]:.4f} |\n')
    f.write('\n')
    
    f.write('### Discussion on Momentum Alphas\n')
    f.write('The momentum alphas are not indicative of managerial skill but rather indicate a risk factor not captured by the models. The momentum effect was documented by Jegadeesh and Titman (1993), and once published, any skilled manager could have implemented this strategy. The persistent alphas across different factor models suggest momentum captures a risk dimension not explained by market, size, value, profitability, or investment factors.\n\n')
    
    # Question 4
    f.write('## Question 4: Betting-Against-Beta Factor\n\n')
    
    f.write('### Mean Returns by Beta Decile\n')
    f.write('| Portfolio | Equal-Weighted | Value-Weighted |\n')
    f.write('|-----------|----------------|----------------|\n')
    for i in range(1, 11):
        f.write(f'| {i} | {mean_bab_returns_ew[f"EW_P{i}"]:.4f} | {mean_bab_returns_vw[f"VW_P{i}"]:.4f} |\n')
    f.write('\n')
    
    f.write('### BAB Portfolio Performance\n')
    f.write('| Metric | Equal-Weighted | Value-Weighted |\n')
    f.write('|--------|----------------|----------------|\n')
    for idx in bab_performance.index:
        f.write(f'| {idx} | {bab_performance["Equal-Weighted BAB"][idx]:.4f} | {bab_performance["Value-Weighted BAB"][idx]:.4f} |\n')
    f.write('\n')
    
    f.write('### Factor Model Alphas\n')
    f.write('| Model | Equal-Weighted | Value-Weighted |\n')
    f.write('|-------|----------------|----------------|\n')
    for idx in bab_factor_models.index:
        if 'Alpha' in idx:
            f.write(f'| {idx} | {bab_factor_models["Equal-Weighted BAB"][idx]:.4f} | {bab_factor_models["Value-Weighted BAB"][idx]:.4f} |\n')
    f.write('\n')
    
    f.write('### Reducing BAB Strategy Volatility\n')
    f.write('To reduce the volatility of the BAB strategy while preserving its alpha, several approaches could be implemented:\n\n')
    f.write('1. **Leverage Adjustment**: Adjust position sizing based on beta estimates. Instead of equal long-short positions, scale positions to have equal beta exposure across the long and short sides.\n\n')
    f.write('2. **Sector Neutrality**: Implement the strategy within sectors to avoid unintended sector bets that might drive volatility.\n\n')
    f.write('3. **Volatility Targeting**: Dynamically adjust the overall portfolio exposure to maintain constant volatility over time.\n\n')
    f.write('4. **Signal Smoothing**: Use longer-term beta estimates or smooth beta signals to reduce turnover and noise in portfolio construction.\n\n')
    f.write('5. **Portfolio Constraints**: Implement tighter constraints on individual position sizes, industry exposures, and other risk factors.\n\n')
    f.write('6. **Blending with Other Factors**: Combine BAB with other low-correlation factors like quality to create a more diversified strategy.\n\n')

print("Analysis complete! Results are saved in the 'results' directory.")
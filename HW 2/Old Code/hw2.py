import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# SECTION 1: Data Cleaning and Summary Statistics
# =============================================================================

def load_crsp_data():
    """
    Load the cleaned CRSP data. (Assumes crsp_cleaned.csv is already filtered by share and exchange.)
    Also converts key columns to numeric and drops rows with missing PRC or RET.
    """
    df = pd.read_csv("crsp_cleaned.csv")
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    # Convert numeric columns; any problematic entries become NaN.
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
    df['PRC'] = pd.to_numeric(df['PRC'], errors='coerce')
    df['SHROUT'] = pd.to_numeric(df['SHROUT'], errors='coerce')
    # Drop rows where essential numeric values are missing.
    df = df.dropna(subset=['PRC', 'RET', 'SHROUT'])
    return df

df_crsp = load_crsp_data()

def plot_listed_firms(df):
    """
    Plot the number of listed firms (unique PERMNO) per month.
    """
    # Group by date and count unique PERMNO values.
    counts = df.groupby('date')['PERMNO'].nunique()
    plt.figure(figsize=(12,6))
    plt.plot(counts.index, counts.values)
    plt.xlabel('Date')
    plt.ylabel('Number of Listed Firms')
    plt.title('Number of Listed Firms per Month')
    plt.show()

plot_listed_firms(df_crsp)

# =============================================================================
# SECTION 2: Replicate Size Strategy
# =============================================================================

def size_strategy(df):
    """
    Compute size portfolios by sorting stocks into deciles based on market capitalization.
    Uses both equal- and value-weighted returns and forms a long-short portfolio 
    (small decile minus big decile).
    """
    # Compute market capitalization = |PRC| * SHROUT.
    df = df.copy()
    df['mktcap'] = df['PRC'].abs() * df['SHROUT']
    
    # Group by date and assign deciles (using pd.qcut) based on market cap.
    # (If there are ties or too few observations, duplicates='drop' avoids errors.)
    df['size_decile'] = df.groupby('date')['mktcap'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    
    # Equal-weighted returns: average return for stocks in each decile.
    eq_portfolios = df.groupby(['date', 'size_decile'])['RET'].mean().unstack()
    
    # Value-weighted returns: weighted average return using market cap.
    vw_portfolios = df.groupby(['date', 'size_decile']).apply(
        lambda x: np.average(x['RET'], weights=x['mktcap'])
    ).unstack()
    
    # Long-short portfolio: long the smallest decile (assumed label 0) minus short the largest (assumed label 9).
    ls_eq = eq_portfolios[0] - eq_portfolios[9]
    ls_vw = vw_portfolios[0] - vw_portfolios[9]
    
    print("Size Strategy - Equal Weighted Long-Short (Small minus Big):")
    print(f"Mean: {ls_eq.mean():.4f}, Volatility: {ls_eq.std():.4f}, Sharpe: {ls_eq.mean()/ls_eq.std():.4f}")
    print("\nSize Strategy - Value Weighted Long-Short (Small minus Big):")
    print(f"Mean: {ls_vw.mean():.4f}, Volatility: {ls_vw.std():.4f}, Sharpe: {ls_vw.mean()/ls_vw.std():.4f}")
    
    return eq_portfolios, vw_portfolios, ls_eq, ls_vw

eq_size, vw_size, ls_eq_size, ls_vw_size = size_strategy(df_crsp)

# =============================================================================
# Load Factor Data (for regression)
# =============================================================================

def load_ff_factors():
    """
    Load the Fama-French three-factor data.
    Expected columns: date (in YYYYMM format), Mkt-RF, SMB, HML, RF.
    """
    ff3 = pd.read_csv("F-F_Research_Data_Factors.CSV")
    ff3 = ff3.rename(columns=lambda x: x.strip())
    # Rename the date column and convert to datetime (assuming YYYYMM)
    ff3.rename(columns={ff3.columns[0]:'date'}, inplace=True)
    ff3['date'] = pd.to_datetime(ff3['date'], format='%Y%m')
    ff3[['Mkt-RF','SMB','HML','RF']] = ff3[['Mkt-RF','SMB','HML','RF']].apply(pd.to_numeric, errors='coerce')
    return ff3

ff3 = load_ff_factors()

def load_ff5_factors():
    """
    Load the Fama-French five-factor data.
    Expected columns: date, Mkt-RF, SMB, HML, RMW, CMA, RF.
    """
    ff5 = pd.read_csv("F-F_Research_Data_5_Factors_2x3.csv")
    ff5 = ff5.rename(columns=lambda x: x.strip())
    ff5.rename(columns={ff5.columns[0]:'date'}, inplace=True)
    ff5['date'] = pd.to_datetime(ff5['date'], format='%Y%m')
    ff5[['Mkt-RF','SMB','HML','RMW','CMA','RF']] = ff5[['Mkt-RF','SMB','HML','RMW','CMA','RF']].apply(pd.to_numeric, errors='coerce')
    return ff5

ff5 = load_ff5_factors()

def load_momentum_factor():
    """
    Load the momentum factor data.
    Expected columns: date (in YYYYMM format), Mom.
    """
    mom = pd.read_csv("F-F_Momentum_Factor.CSV")
    mom = mom.rename(columns=lambda x: x.strip())
    mom.rename(columns={mom.columns[0]:'date'}, inplace=True)
    mom['date'] = pd.to_datetime(mom['date'], format='%Y%m')
    mom[['Mom']] = mom[['Mom']].apply(pd.to_numeric, errors='coerce')
    return mom

mom = load_momentum_factor()

# =============================================================================
# Regression Helper Function
# =============================================================================

def run_regression(portfolio_returns, factor_data, model_type="CAPM", portfolio_name="Portfolio"):
    # Convert portfolio_returns to a DataFrame and extract the year-month
    df = pd.DataFrame({'ret': portfolio_returns})
    df = df.reset_index()  # assumes the index is date
    df['yearmonth'] = df['date'].dt.to_period('M').astype(str)
    
    # Ensure factor_data has a 'yearmonth' column
    factor_data = factor_data.copy()
    if 'yearmonth' not in factor_data.columns:
        factor_data['yearmonth'] = factor_data['date'].dt.to_period('M').astype(str)
    
    if model_type=="CAPM":
        factors = factor_data[['yearmonth','Mkt-RF','RF']]
        df = pd.merge(df, factors, on='yearmonth', how='inner')
        df['ex_ret'] = df['ret'] - df['RF']
        X = sm.add_constant(df['Mkt-RF'])
        model = sm.OLS(df['ex_ret'], X).fit()
        print(f"\n{portfolio_name} CAPM Regression Results:")
        print(model.summary())
    elif model_type=="FF3":
        factors = factor_data[['yearmonth','Mkt-RF','SMB','HML','RF']]
        df = pd.merge(df, factors, on='yearmonth', how='inner')
        df['ex_ret'] = df['ret'] - df['RF']
        X = df[['Mkt-RF','SMB','HML']]
        X = sm.add_constant(X)
        model = sm.OLS(df['ex_ret'], X).fit()
        print(f"\n{portfolio_name} FF3 Regression Results:")
        print(model.summary())
    elif model_type=="FF5":
        factors = factor_data[['yearmonth','Mkt-RF','SMB','HML','RMW','CMA','RF']]
        df = pd.merge(df, factors, on='yearmonth', how='inner')
        df['ex_ret'] = df['ret'] - df['RF']
        X = df[['Mkt-RF','SMB','HML','RMW','CMA']]
        X = sm.add_constant(X)
        model = sm.OLS(df['ex_ret'], X).fit()
        print(f"\n{portfolio_name} FF5 Regression Results:")
        print(model.summary())
    elif model_type=="FF5+Momentum":
        # Ensure the momentum factor data also has a 'yearmonth' column.
        global mom
        mom = mom.copy()
        if 'yearmonth' not in mom.columns:
            mom['yearmonth'] = mom['date'].dt.to_period('M').astype(str)
        # Merge FF5 factor data with momentum factor.
        factors = pd.merge(factor_data, mom, on='yearmonth', how='inner')
        factors = factors[['yearmonth','Mkt-RF','SMB','HML','RMW','CMA','RF','Mom']]
        df = pd.merge(df, factors, on='yearmonth', how='inner')
        df['ex_ret'] = df['ret'] - df['RF']
        X = df[['Mkt-RF','SMB','HML','RMW','CMA','Mom']]
        X = sm.add_constant(X)
        model = sm.OLS(df['ex_ret'], X).fit()
        print(f"\n{portfolio_name} FF5+Momentum Regression Results:")
        print(model.summary())
    return

# Run regressions on the long-short portfolios from the size strategy.
print("=== Size Strategy Regressions ===")
run_regression(ls_eq_size, ff3, model_type="CAPM", portfolio_name="Size LS Equal-Weighted")
run_regression(ls_eq_size, ff3, model_type="FF3", portfolio_name="Size LS Equal-Weighted")
run_regression(ls_vw_size, ff3, model_type="CAPM", portfolio_name="Size LS Value-Weighted")
run_regression(ls_vw_size, ff3, model_type="FF3", portfolio_name="Size LS Value-Weighted")

# Subperiod analysis: post-Fama French (June 1992) and post-Dot-Com (around 2002)
def analyze_subperiod(portfolio_returns, label="Size LS Portfolio"):
    df = portfolio_returns.to_frame(name='ret').reset_index()
    df['year'] = df['date'].dt.year
    overall_stats = {'Mean': df['ret'].mean(), 'Volatility': df['ret'].std(), 'Sharpe': df['ret'].mean()/df['ret'].std()}
    post1992 = df[df['date'] >= pd.to_datetime("1992-06-01")]
    post2002 = df[df['date'] >= pd.to_datetime("2002-01-01")]
    stats_1992 = {'Mean': post1992['ret'].mean(), 'Volatility': post1992['ret'].std(), 'Sharpe': post1992['ret'].mean()/post1992['ret'].std()}
    stats_2002 = {'Mean': post2002['ret'].mean(), 'Volatility': post2002['ret'].std(), 'Sharpe': post2002['ret'].mean()/post2002['ret'].std()}
    
    print(f"\nSubperiod Analysis for {label}:")
    print("Overall:", overall_stats)
    print("Post June 1992:", stats_1992)
    print("Post January 2002:", stats_2002)

analyze_subperiod(ls_eq_size, label="Size LS Equal-Weighted")
analyze_subperiod(ls_vw_size, label="Size LS Value-Weighted")

# =============================================================================
# SECTION 3: Replicate Momentum Strategy
# =============================================================================

def momentum_strategy(df):
    """
    Calculate momentum portfolios based on the cumulative return over an 11-month window
    (from t-12 to t-1) for each stock. Form deciles and compute equal- and value-weighted 
    portfolio returns and the long-short (winners minus losers) portfolio.
    """
    df = df.copy().sort_values(['PERMNO', 'date'])
    # Calculate cumulative return for an 11-month window (shifted by one to exclude current month)
    df['cum_ret'] = df.groupby('PERMNO')['RET'].transform(
        lambda x: x.shift(1).rolling(window=11, min_periods=11).apply(lambda r: np.prod(1 + r) - 1, raw=True)
    )
    # Drop stocks with incomplete momentum history.
    df_mom = df.dropna(subset=['cum_ret']).copy()
    
    # Assign deciles based on momentum (cumulative return) for each month.
    df_mom['mom_decile'] = df_mom.groupby('date')['cum_ret'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    
    # Equal-weighted momentum portfolios.
    eq_mom = df_mom.groupby(['date', 'mom_decile'])['RET'].mean().unstack()
    
    # Value-weighted momentum portfolios.
    df_mom['mktcap'] = df_mom['PRC'].abs() * df_mom['SHROUT']
    vw_mom = df_mom.groupby(['date', 'mom_decile']).apply(
        lambda x: np.average(x['RET'], weights=x['mktcap'])
    ).unstack()
    
    # Long-short portfolio: winners (decile 9) minus losers (decile 0)
    ls_eq_mom = eq_mom[9] - eq_mom[0]
    ls_vw_mom = vw_mom[9] - vw_mom[0]
    
    print("\nMomentum Strategy - Equal Weighted Long-Short (Winners minus Losers):")
    print(f"Mean: {ls_eq_mom.mean():.4f}, Volatility: {ls_eq_mom.std():.4f}, Sharpe: {ls_eq_mom.mean()/ls_eq_mom.std():.4f}")
    print("\nMomentum Strategy - Value Weighted Long-Short (Winners minus Losers):")
    print(f"Mean: {ls_vw_mom.mean():.4f}, Volatility: {ls_vw_mom.std():.4f}, Sharpe: {ls_vw_mom.mean()/ls_vw_mom.std():.4f}")
    
    return eq_mom, vw_mom, ls_eq_mom, ls_vw_mom

eq_mom, vw_mom, ls_eq_mom, ls_vw_mom = momentum_strategy(df_crsp)

# Run regressions for momentum strategy long-short portfolios.
print("\n=== Momentum Strategy Regressions ===")
run_regression(ls_eq_mom, ff3, model_type="CAPM", portfolio_name="Momentum LS Equal-Weighted")
run_regression(ls_eq_mom, ff3, model_type="FF3", portfolio_name="Momentum LS Equal-Weighted")
run_regression(ls_eq_mom, ff5, model_type="FF5", portfolio_name="Momentum LS Equal-Weighted")

print("\nMomentum Strategy - Value Weighted Regressions:")
run_regression(ls_vw_mom, ff3, model_type="CAPM", portfolio_name="Momentum LS Value-Weighted")
run_regression(ls_vw_mom, ff3, model_type="FF3", portfolio_name="Momentum LS Value-Weighted")
run_regression(ls_vw_mom, ff5, model_type="FF5", portfolio_name="Momentum LS Value-Weighted")

# =============================================================================
# SECTION 4: Replicate Betting-Against-Beta (BAB) Strategy
# =============================================================================

def calculate_bab_beta(df, ff_factors):
    """
    For each stock, calculate the rolling CAPM beta using a 36-month window.
    This version aligns CRSP and factor data on a common year-month key.
    """
    df = df.copy()
    ff_factors = ff_factors.copy()
    
    # Create a common "yearmonth" key for both dataframes.
    df['yearmonth'] = df['date'].dt.to_period('M').dt.to_timestamp()
    ff_factors['yearmonth'] = ff_factors['date'].dt.to_period('M').dt.to_timestamp()
    
    # Merge on the yearmonth column.
    df_merged = pd.merge(df, ff_factors[['yearmonth', 'Mkt-RF', 'RF']], on='yearmonth', how='inner')
    
    # Calculate excess returns.
    df_merged['ex_ret'] = df_merged['RET'] - df_merged['RF']
    
    def rolling_beta(group):
        group = group.sort_values('yearmonth')
        if len(group) < 36:
            group['beta'] = np.nan
            return group
        # Prepare regression variables.
        X = group['Mkt-RF'].values
        y = group['ex_ret'].values
        X_const = sm.add_constant(X)
        rols = RollingOLS(y, X_const, window=36)
        rres = rols.fit()
        # Directly assign the beta estimates.
        group['beta'] = rres.params[:, 1]
        return group
    
    df_beta = df_merged.groupby('PERMNO').apply(rolling_beta)
    return df_beta

# Now use the updated function:
df_bab = calculate_bab_beta(df_crsp, ff3)

if df_bab.empty:
    print("The merged dataframe for BAB strategy is empty. Check date ranges and formats in your CRSP and factor data.")
else:
    # Only drop rows if beta column exists and dataframe is not empty.
    df_bab = df_bab.dropna(subset=['beta'])

def bab_strategy(df):
    """
    Form BAB portfolios by sorting stocks into deciles based on the estimated beta.
    Compute equal- and value-weighted returns and form a long-short portfolio (low beta minus high beta).
    """
    df = df.copy()
    df['bab_decile'] = df.groupby('date')['beta'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    
    # Equal-weighted BAB portfolios.
    eq_bab = df.groupby(['date', 'bab_decile'])['RET'].mean().unstack()
    
    # Value-weighted BAB portfolios.
    df['mktcap'] = df['PRC'].abs() * df['SHROUT']
    vw_bab = df.groupby(['date', 'bab_decile']).apply(
        lambda x: np.average(x['RET'], weights=x['mktcap'])
    ).unstack()
    
    # Long-short BAB portfolio: low beta (decile 0) minus high beta (decile 9)
    ls_eq_bab = eq_bab[0] - eq_bab[9]
    ls_vw_bab = vw_bab[0] - vw_bab[9]
    
    print("\nBAB Strategy - Equal Weighted Long-Short (Low Beta minus High Beta):")
    print(f"Mean: {ls_eq_bab.mean():.4f}, Volatility: {ls_eq_bab.std():.4f}, Sharpe: {ls_eq_bab.mean()/ls_eq_bab.std():.4f}")
    print("\nBAB Strategy - Value Weighted Long-Short (Low Beta minus High Beta):")
    print(f"Mean: {ls_vw_bab.mean():.4f}, Volatility: {ls_vw_bab.std():.4f}, Sharpe: {ls_vw_bab.mean()/ls_vw_bab.std():.4f}")
    
    return eq_bab, vw_bab, ls_eq_bab, ls_vw_bab

eq_bab, vw_bab, ls_eq_bab, ls_vw_bab = bab_strategy(df_bab)

# Run regressions for BAB long-short portfolios.
print("\n=== BAB Strategy Regressions ===")
run_regression(ls_eq_bab, ff3, model_type="CAPM", portfolio_name="BAB LS Equal-Weighted")
run_regression(ls_eq_bab, ff3, model_type="FF3", portfolio_name="BAB LS Equal-Weighted")
run_regression(ls_eq_bab, ff5, model_type="FF5", portfolio_name="BAB LS Equal-Weighted")
run_regression(ls_eq_bab, ff5, model_type="FF5+Momentum", portfolio_name="BAB LS Equal-Weighted")

print("\nBAB Strategy - Value Weighted Regressions:")
run_regression(ls_vw_bab, ff3, model_type="CAPM", portfolio_name="BAB LS Value-Weighted")
run_regression(ls_vw_bab, ff3, model_type="FF3", portfolio_name="BAB LS Value-Weighted")
run_regression(ls_vw_bab, ff5, model_type="FF5", portfolio_name="BAB LS Value-Weighted")
run_regression(ls_vw_bab, ff5, model_type="FF5+Momentum", portfolio_name="BAB LS Value-Weighted")

# =============================================================================
# Question 4(e): Volatility Reduction Discussion
# =============================================================================
print("\nFor the BAB strategy, if tasked with reducing volatility one might consider applying volatility scaling or risk parity methods, or implementing filters to remove stocks with extremely high beta estimates, thus reducing the overall portfolio volatility.")

# End of Script
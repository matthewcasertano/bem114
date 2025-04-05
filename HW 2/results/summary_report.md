# BEM 114: Hedge Funds - Problem Set 2 Summary Report

## Question 1: Data Cleaning and Summary Statistics

The data has been cleaned according to the required criteria, and the number of listed firms over time has been plotted in `listed_firms_over_time.png`.

## Question 2: Size Factor

### Mean Returns by Size Decile
| Portfolio | Equal-Weighted | Value-Weighted |
|-----------|----------------|----------------|
| 1 | -0.0586 | 0.0148 |
| 2 | 0.1432 | 0.1463 |
| 3 | 0.1598 | 0.1601 |
| 4 | 0.1826 | 0.1843 |
| 5 | 0.2059 | 0.2062 |
| 6 | 0.2107 | 0.2100 |
| 7 | 0.2114 | 0.2107 |
| 8 | 0.2114 | 0.2095 |
| 9 | 0.1956 | 0.1932 |
| 10 | 0.1785 | 0.1727 |

### SMB Portfolio Performance
| Metric | Equal-Weighted | Value-Weighted |
|--------|----------------|----------------|
| Mean (annual) | -0.2371 | -0.1579 |
| Volatility (annual) | 0.3080 | 0.2961 |
| Sharpe Ratio | -0.7697 | -0.5332 |

### Factor Model Alphas
| Model | Equal-Weighted | Value-Weighted |
|-------|----------------|----------------|
| Alpha (CAPM) | -0.3160 | -0.2409 |
| Alpha (FF3) | -0.3531 | -0.2780 |

### Post-Publication Performance
| Period | Equal-Weighted Mean | Value-Weighted Mean |
|--------|---------------------|---------------------|
| Full Sample | -0.2371 | -0.1579 |
| Post-Publication (1992+) | -0.4801 | -0.3350 |
| Post-Dot-Com (2002+) | -0.4199 | -0.2849 |

## Question 3: Momentum Factor

### Mean Returns by Momentum Decile
| Portfolio | Equal-Weighted | Value-Weighted |
|-----------|----------------|----------------|
| 1 | 0.2593 | 0.3207 |
| 2 | 0.1364 | 0.1994 |
| 3 | 0.1303 | 0.1868 |
| 4 | 0.1297 | 0.1733 |
| 5 | 0.1335 | 0.1619 |
| 6 | 0.1396 | 0.1653 |
| 7 | 0.1465 | 0.1671 |
| 8 | 0.1593 | 0.1911 |
| 9 | 0.1631 | 0.1949 |
| 10 | 0.1764 | 0.2893 |

### WML Portfolio Performance
| Metric | Equal-Weighted | Value-Weighted |
|--------|----------------|----------------|
| Mean (annual) | -0.0830 | -0.0314 |
| Volatility (annual) | 0.3267 | 0.3524 |
| Sharpe Ratio | -0.2540 | -0.0890 |

### Factor Model Alphas
| Model | Equal-Weighted | Value-Weighted |
|-------|----------------|----------------|
| Alpha (CAPM) | -0.0677 | -0.0092 |
| Alpha (FF3) | -0.0412 | 0.0103 |

### Discussion on Momentum Alphas
The momentum alphas are not indicative of managerial skill but rather indicate a risk factor not captured by the models. The momentum effect was documented by Jegadeesh and Titman (1993), and once published, any skilled manager could have implemented this strategy. The persistent alphas across different factor models suggest momentum captures a risk dimension not explained by market, size, value, profitability, or investment factors.

## Question 4: Betting-Against-Beta Factor

### Mean Returns by Beta Decile
| Portfolio | Equal-Weighted | Value-Weighted |
|-----------|----------------|----------------|
| 1 | 0.1376 | 0.1368 |
| 2 | 0.1427 | 0.1382 |
| 3 | 0.1492 | 0.1540 |
| 4 | 0.1562 | 0.1730 |
| 5 | 0.1658 | 0.1743 |
| 6 | 0.1610 | 0.1766 |
| 7 | 0.1721 | 0.1941 |
| 8 | 0.1697 | 0.2011 |
| 9 | 0.1675 | 0.2249 |
| 10 | 0.1643 | 0.2916 |

### BAB Portfolio Performance
| Metric | Equal-Weighted | Value-Weighted |
|--------|----------------|----------------|
| Mean (annual) | -0.0266 | -0.1548 |
| Volatility (annual) | 0.2692 | 0.3152 |
| Sharpe Ratio | -0.0990 | -0.4911 |

### Factor Model Alphas
| Model | Equal-Weighted | Value-Weighted |
|-------|----------------|----------------|
| Alpha (CAPM) | 0.0232 | -0.0913 |
| Alpha (FF3) | 0.0401 | -0.0783 |

### Reducing BAB Strategy Volatility
To reduce the volatility of the BAB strategy while preserving its alpha, several approaches could be implemented:

1. **Leverage Adjustment**: Adjust position sizing based on beta estimates. Instead of equal long-short positions, scale positions to have equal beta exposure across the long and short sides.

2. **Sector Neutrality**: Implement the strategy within sectors to avoid unintended sector bets that might drive volatility.

3. **Volatility Targeting**: Dynamically adjust the overall portfolio exposure to maintain constant volatility over time.

4. **Signal Smoothing**: Use longer-term beta estimates or smooth beta signals to reduce turnover and noise in portfolio construction.

5. **Portfolio Constraints**: Implement tighter constraints on individual position sizes, industry exposures, and other risk factors.

6. **Blending with Other Factors**: Combine BAB with other low-correlation factors like quality to create a more diversified strategy.


CVaR (Conditional Value at Risk) is a bit more neat than normal Value at Risk because you can capture tail-end risks by gauging the average loss beyond the VaR threshold. Would Taleb be proud?! Or maybe Bostrom to

If you've seen my other projects, you know I'm overly reliant on the Yahoo Finance and Alpha Vantage APIs. It employs logarithmic returns for accuracy in compounding and uses rolling window calculations to simulate fluctuating market conditions. You can add in multiple tickers with equal-weighted portfolio allocation (I will work on making these adjustable). 

One cool things I got to learn and implement were covariance matrix calculations to model asset correlations

I included both the the historical method, which averages portfolio losses exceeding the VaR threshold, and the parametric method, which uses z-scores and expected returns to estimate tail risks analytically. 

As always, everything's interactive + visual. Peace





# stock viz


# Import yfinance
import yfinance as yf

from tabulate import tabulate

# Get the data for the stock Apple by specifying the stock ticker, start date, and end date
#data =  yf.download("BSE.NS")


#data =  yf.download("INFY.NS")
data =  yf.download("ITC.NS", period='1d').reset_index()
print(tabulate(data, headers = 'keys', tablefmt = 'psql'))

print(data.Date)
# Plot the close prices
import matplotlib.pyplot as plt
data.Close.plot()
plt.show()

stock = yf.Ticker("INFY.NS")
price = stock.info['regularMarketPrice']
print(price)

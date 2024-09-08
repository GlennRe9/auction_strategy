<h><b>Bond Auction Strategy and Yield Analysis</b></h>

This project implements three strategies for bond auctions, analyzing bond yields, prices, and future strategies based on French bond auction data.
We present two naive strategies and one more refined relative value strategy.


<b>Features</b>
Bond Data Cleaning: Functions to clean and preprocess raw auction and bond history data.</n>

Auction Strategy: Implements strategies around bond auctions, including long strategies and simple auction strategies.
Yield and Price Analysis: Includes tools to analyze bond yields, calculate the time to maturity, and create maturity buckets.
Future Strategy: Simulates a trading strategy involving future trades to manage duration exposure.
Benchmark Creation: Calculates performance of French benchmark bonds in a simulated portfolio.

Installation
To run this project, you will need to install the following libraries:


```markdown
pip install pandas numpy matplotlib seaborn
```
Ensure that you have Python 3.x installed on your machine.
Clone this repository:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

<b>Prepare the input data files:</b>

Futures dataset.xlsx: Contains future data for benchmark calculation.
bond_data.xlsx: Contains bond metadata (e.g., ISIN, maturity).
bond_hist_data.xlsx: Contains historical bond price and yield data.
auction_hist.xlsx: Contains historical auction data.
Place these files in the root directory of the project.

<b>Run the main script:</b>

```bash
python main.py
```
The script will process the data, run several bond auction strategies, and output performance results, including visual plots of cumulative profit and loss (PnL) from different strategies.


<b>Strategies</b>

Simple Auction Strategy: Sells bonds on the Friday preceding the auction and buys them back on auction day.
Future Strategy: Trades futures to minimize duration exposure, buying futures on the Friday preceding the auction and selling them back on the auction day.
Auction Strategy: A more complex strategy that calculates the probability of bond auctions and implements trades accordingly.

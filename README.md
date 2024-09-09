<h><b>Bond Auction Strategy and Yield Analysis</b></h>

This project implements three strategies for bond auctions on French bond auction data.
We present two naive strategies and one more refined relative value strategy.

<b>Strategies</b>
The naive strategy consists of selling the bond on the Friday of the announcement vs a near bond and closing the position on the auction day
The future strategy consists of selling the 10-year bond on the Friday of the announcement vs the 10-year OAT future, and closing the position on auction day
The RV strategy, our main strategym consists of opening a position on the Monday before the announcement and closing the position on the evening of the announcement. For
each pre-announcement day we forecast the most likely bonds to be auctioned, given the cash price relative to cash prices in that maturity bucket and given the current issuance of
those bonds relative to the average issuance in the maturity bucket.

<b>Relative Value Strategy</b>
• Based on the assumption that the bonds that have the highest probability of being tapped are those which have a lower relative issuance and higher relative cash price 
• ‘Relative’ means within its maturity bucket – e.g. a 20-year bond will be more likely to be tapped if, within its maturity bucket, it has a higher cash price than its peers and a lower amount outstanding than its peers
• Each Monday before the long term auction announcement I analyse the previous two years and select 3 bonds according to the above methodology  
• On that Monday evening I implement the trade: I sell (or rather, in a benchmarked portfolio, don’t buy) the bond vs I buy two adjacent bonds (or just one when two are not present). Therefore, each Monday before the announcement I am operating either a butterfly or a curve strategy which I close on the evening of the Friday announcement, at closing prices

<b>Features</b>
Bond Data Cleaning: Functions to clean and preprocess raw auction and bond history data.</n>

Auction Strategy: Implements strategies around bond auctions, including long/ strategies and simple auction strategies.
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


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import logging
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
from datetime import datetime
from pandas.tseries.offsets import BDay



month_mapping = {
    'janvier': 'January',
    'février': 'February',
    'mars': 'March',
    'avril': 'April',
    'mai': 'May',
    'juin': 'June',
    'juillet': 'July',
    'août': 'August',
    'septembre': 'September',
    'octobre': 'October',
    'novembre': 'November',
    'décembre': 'December'
}
bins = [0, 3, 5, 7, 10, 12, 15, 17, 20, 25, 30, np.inf]
names = ['0-3', '3-5', '5-7', '7-10', '10-12', '12-15', '15-17', '17-20', '20-25', '25-30', '30+']

def clean_future_data(future_data):
    # Start from row 3
    future_data.columns = future_data.iloc[2]
    future_data = future_data.iloc[5:]
    future_data.rename(columns={future_data.columns[0]: 'date'}, inplace=True)
    # Tranforming excel dates to pandas dates
    #future_data['date'] = pd.TimedeltaIndex(future_data['date'], unit='d') + pd.Timestamp('1899-12-30')
    future_data['date'] = pd.to_datetime(future_data['date'], format='%d/%m/%Y')
    #future_data = future_data.set_index('date')
    return future_data

def clean_auction_hist(auction_hist):
    # Remove French column names
    auction_hist.columns = auction_hist.columns.str.split('\n').str[1]
    # Remove inflation linked bonds
    auction_hist = auction_hist[auction_hist['type of auction'] != 'adju_I']
    # I decide to start looking at auction histories from 2012 onwards
    auction_hist = auction_hist[auction_hist['auction date'] >= '2012-01-01']
    # Remove problematic bonds present in this list ['FR0010466938'] from ISIN code column
    #problematic_isins = ['FR0010466938', 'FR0011317783', 'FR0011486067', 'FR0013451507', 'FR0013516549', 'FR0014002WK3', 'FR0014002JM6', 'FR0014007L00', 'FR0014009O62']
    #auction_hist = auction_hist[~auction_hist['ISIN code'].isin(problematic_isins)]
    # Checking
    auction_hist = auction_hist.sort_values(by=['auction date', 'ISIN code'])
    auction_hist['amount outstanding'] = auction_hist.groupby('ISIN code')['total amount issued'].cumsum()
    auction_hist['ratio'] = auction_hist['total amount issued'] / auction_hist['amount outstanding']
    # Getting maturity for each bond
    auction_hist['maturity'] = auction_hist['line'].apply(lambda text: ' '.join(text.split()[-3:]))
    auction_hist['maturity'] = auction_hist['maturity'].replace(month_mapping, regex=True)
    auction_hist['maturity'] = pd.to_datetime(auction_hist['maturity'], format='%d %B %Y')
    # Calculate time to maturity in years
    auction_hist['ttm'] = (auction_hist['maturity'] - auction_hist['auction date']) / np.timedelta64(1, 'Y')
    # Map each maturity into maturity buckets: 0-3, 3-5, 5-7, 7-10, 10-15, 15-20, 20-25, 25-30, 30+
    auction_hist['maturity bucket'] = pd.cut(auction_hist['ttm'], bins, labels=names)

    # I calculate averages per bucket based on the amounts outstanding per bucket at that point in time
    auction_hist = bucket_average_calculator(auction_hist)
    # I now calculate a ratio which tells us how much outstanding that bond has relative to the agerage in its bucket
    auction_hist['outstanding ratio'] = auction_hist['amount outstanding'] / auction_hist['bucket average']
    return auction_hist

def bucket_average_calculator(auction_hist):
    auction_hist['cummax_amount_outs'] = auction_hist.groupby('ISIN code')['amount outstanding'].cummax()
    auction_hist['bucket average'] = np.nan
    for date in auction_hist['auction date'].unique():
        # Filter the data for all entries up to the current date
        temp_df = auction_hist[auction_hist['auction date'] <= date]
        # Find the max amount outstanding for each ISIN up to this date
        max_per_isin = temp_df.groupby('ISIN code')['cummax_amount_outs'].max()
        # merge into max_per isin the maturity bucket on the isin code
        max_per_isin = max_per_isin.reset_index().merge(temp_df[['auction date', 'ISIN code', 'maturity bucket']],
                                                        on='ISIN code')
        # Remove duplicate ISINS
        max_per_isin = max_per_isin.drop_duplicates(subset=['ISIN code'], keep='last')
        # Calculate the average amount outstanding for each maturity bucket according to the average of the last 2 years
        most_recent_date = max_per_isin['auction date'].max()
        start_date = most_recent_date - pd.Timedelta(days=720)
        last_2year_data = max_per_isin[(max_per_isin['auction date'] >= start_date)]
        averages = last_2year_data.groupby('maturity bucket')['cummax_amount_outs'].mean().reset_index()
        # Use this version of averages only if you want a full mean of the whole sample
        # averages = max_per_isin.groupby('maturity bucket')['cummax_amount_outs'].mean().reset_index()
        # Rename cummax_amount_outs to maturity bucket average
        averages = averages.rename(columns={'cummax_amount_outs': 'bucket average'})
        # Merge cummax_amount_outs from averages into max_per_isin on the maturity bucket
        max_per_isin = max_per_isin.merge(averages, on='maturity bucket', how='left')
        # Merge bucket_average from max_per_isin into temp_df on the isin code
        auction_hist = auction_hist.merge(max_per_isin[['ISIN code', 'auction date', 'bucket average']],
                                          on=['ISIN code', 'auction date'], how='left', suffixes=('', '_new'))
        auction_hist['bucket average'] = auction_hist.apply(
            lambda row: row['bucket average'] if pd.notna(row['bucket average']) else row['bucket average_new'], axis=1
        )
        auction_hist = auction_hist.drop(columns=['bucket average_new'])
    return auction_hist
def clean_bond_hist(bond_yield_hist, bond_data):
    bond_yield_hist.rename(columns={bond_yield_hist.columns[0]: 'date'}, inplace=True)
    bond_yield_hist['date'] = pd.to_datetime(bond_yield_hist['date'], format='%d-%m-%Y')
    # Remove columns where all values are NaN
    bond_yield_hist = bond_yield_hist.dropna(axis=1, how='all')
    # Remove problematic bonds with isin in ['FR0011317783']
    bond_yield_hist = bond_yield_hist.drop(columns=['FR0011317783'])
    # Remove columns which we didn't manage to download
    bond_yield_hist = bond_yield_hist.select_dtypes(include=[np.number]).join(bond_yield_hist[['date']])
    #bond_yield_hist = bond_yield_hist.set_index('date')
    # Drop duplicate rows
    bond_yield_hist = bond_yield_hist.drop_duplicates(subset=['date'], keep='first')

    long_format = bond_yield_hist.melt(id_vars=['date'], var_name='ISIN', value_name='Yield')
    long_format = long_format.merge(bond_data[['ISIN', 'Maturity']], on='ISIN')
    long_format['date'] = pd.to_datetime(long_format['date'])
    pivoted_df = long_format.pivot(index='date', columns='ISIN', values='Yield')
    maturity_row = long_format.groupby('ISIN')['Maturity'].first().to_frame().T
    maturity_row.index = ['Maturity']
    bond_yield_hist = pd.concat([maturity_row, pivoted_df])

    today = pd.Timestamp(datetime.now().date())
    maturity_dates = bond_yield_hist.loc['Maturity']
    ttm = (today - maturity_dates) / np.timedelta64(1, 'Y') * (-1)
    maturity_buckets = pd.cut(ttm, bins=bins, labels=names, right=False)
    # Set the maturity bucket after maturity row
    df_part1 = bond_yield_hist.loc[:'Maturity']
    df_part2 = bond_yield_hist.loc['Maturity':]
    df_part1.loc['maturity bucket'] = maturity_buckets
    final_df = pd.concat([df_part1, df_part2.iloc[1:]])

    return final_df

def update_bond_hist_maturity_bucket(bond_yield_hist, current_day):
    maturity_dates = bond_yield_hist.loc['Maturity']
    ttm = (current_day - maturity_dates) / np.timedelta64(1, 'Y') * (-1)
    maturity_buckets = pd.cut(ttm, bins=bins, labels=names, right=False)
    # remove maturity bucket row
    bond_yield_hist = bond_yield_hist.drop('maturity bucket')
    #bond_yield_hist = bond_yield_hist.iloc[:-1]
    # Set the maturity bucket after maturity row
    df_part1 = bond_yield_hist.loc[:'Maturity']
    df_part2 = bond_yield_hist.loc['Maturity':]
    df_part1.loc['maturity bucket'] = maturity_buckets
    final_df = pd.concat([df_part1, df_part2.iloc[1:]])
    return final_df
def clean_bond_price_hist(bond_price_hist):
    bond_price_hist.columns = bond_price_hist.columns.str[:-1]
    bond_price_hist.rename(columns={bond_price_hist.columns[0]: 'date'}, inplace=True)
    bond_price_hist = bond_price_hist.iloc[1:]
    bond_price_hist['date'] = pd.to_datetime(bond_price_hist['date'])
    bond_price_hist['date'] = bond_price_hist['date'].dt.strftime('%Y-%m-%d')
    bond_price_hist = bond_price_hist.sort_values(by='date')

    return bond_price_hist

def study_bond_price_hist(bond_price_hist, bond_data):

    long_format = bond_price_hist.melt(id_vars=['date'], var_name='ISIN', value_name='Price')
    long_format = long_format.merge(bond_data[['ISIN', 'Maturity']], on='ISIN')
    long_format['date'] = pd.to_datetime(long_format['date'])
    pivoted_df = long_format.pivot(index='date', columns='ISIN', values='Price')
    maturity_row = long_format.groupby('ISIN')['Maturity'].first().to_frame().T
    maturity_row.index = ['Maturity']
    bond_price_hist = pd.concat([maturity_row, pivoted_df])

    # We now calculate a relative price for each bond, where the relative price is the bond's cash price
    # relative to the average prices of the bucket that bond belongs in at that time
    stats_format = long_format.copy()
    stats_format['ttm'] = (long_format['Maturity'] - long_format['date']) / np.timedelta64(1, 'Y')
    stats_format['maturity bucket'] = pd.cut(stats_format['ttm'], bins, labels=names)

    stats_format.sort_values(by=['date']).reset_index()
    stats_format['bucket price average'] = np.nan
    # What we do here is the same logic as the average issuance per bucket calculation, except we do this for prices
    #for index, row in stats_format.iterrows():
    # Uncomment loop below to recalculate average prices - I am commenting because it takes two hours to run
    # for date in stats_format['date'].unique():
    #     # remove bucket price average column from stats_format
    #     temp_df = stats_format[stats_format['date'] <= date]
    #     temp_df = temp_df.drop(columns=['bucket price average'])
    #     most_recent_date = temp_df['date'].max()
    #     start_date = most_recent_date - pd.Timedelta(days=365)
    #     filtered_df = temp_df[(temp_df['date'] >= start_date)]
    #     averages = filtered_df.groupby('maturity bucket')['Price'].mean().reset_index()
    #     averages = averages.rename(columns={'Price': 'bucket price average'})
    #     filtered_df = filtered_df.merge(averages, on='maturity bucket', how='left')
    #     stats_format = stats_format.merge(filtered_df[['ISIN', 'date', 'bucket price average']],on=['ISIN', 'date'], how='left', suffixes=('', '_new'))
    #     stats_format['bucket price average'] = stats_format.apply(lambda row: row['bucket price average'] if pd.notna(row['bucket price average']) else row['bucket price average_new'], axis=1)
    #     stats_format = stats_format.drop(columns=['bucket price average_new'])

    #stats_format['relative price'] = stats_format['Price'] / stats_format['bucket price average']
    stats_format = pd.read_csv('price_stats.csv', index_col=0, parse_dates=True)
    stats_format['date'] = pd.to_datetime(stats_format['date'])

    return stats_format, bond_price_hist

def benchmark_creation(future_data):
    # We calculate yield changes for the 4 French benchmark bonds
    dur_2y = 1.9
    dur_5y = 4.3
    dur_10y = 9
    dur_30y = 26
    rel_dur_2y = 0.013 * dur_2y
    rel_dur_5y = 0.048 * dur_5y
    rel_dur_10y = 0.083 * dur_10y
    rel_dur_30y = 0.058 * dur_30y
    # Calculating yield changes for the 4 French benchmark bonds
    future_data['2y_fr_change'] = future_data['GTFRF2Y Govt'] - future_data['GTFRF2Y Govt'].shift(1)
    future_data['5y_fr_change'] = future_data['GTFRF5Y Govt'] - future_data['GTFRF5Y Govt'].shift(1)
    future_data['10y_fr_change'] = future_data['GTFRF10Y Govt'] - future_data['GTFRF10Y Govt'].shift(1)
    future_data['30y_fr_change'] = future_data['GTFRF30Y Govt'] - future_data['GTFRF30Y Govt'].shift(1)
    # Calculating the contribution of each bond to the portfolio return
    future_data['ret_contr_2y'] = -rel_dur_2y * future_data['2y_fr_change']
    future_data['ret_contr_5y'] = -rel_dur_5y * future_data['5y_fr_change']
    future_data['ret_contr_10y'] = -rel_dur_10y * future_data['10y_fr_change']
    future_data['ret_contr_30y'] = -rel_dur_30y * future_data['30y_fr_change']
    # Calculating the portfolio return contribution of the French bonds
    future_data['fr_benchmark_ret'] = future_data['ret_contr_2y'] + future_data['ret_contr_5y'] + future_data['ret_contr_10y'] + future_data['ret_contr_30y']
    return future_data
    # If we want to plot the benchmark return
    # data_2023 = future_data.tail(265)
    # data_2023['cum_sum'] = data_2023['fr_benchmark_ret'].cumsum()
    # data_2023 = data_2023.reset_index()
    # data_2023.plot(x='date', y='cum_sum', kind='line')

def auction_des_stats(auction_hist):
    breakpoint()

def extract_yield(bond_yield_hist, implementation_date, isin):
    '''
    This function is used to extract the yield of a bond at a given date. If the yield is not available at that date,
    it will return the yield of the next available date.
    '''
    try:
        value = bond_yield_hist.loc[implementation_date, isin]
        if np.isnan(value):
            cleaned_index = pd.to_datetime(bond_yield_hist.index, errors='coerce').dropna()
            next_available_dates = cleaned_index[cleaned_index >= implementation_date]
            if not next_available_dates.empty:
                next_date = next_available_dates[1]
                value = bond_yield_hist.loc[next_date, isin]
            else:
                value = None
    except KeyError:
        cleaned_index = pd.to_datetime(bond_yield_hist.index, errors='coerce').dropna()
        next_available_dates = cleaned_index[cleaned_index >= implementation_date]
        if not next_available_dates.empty:
            next_date = next_available_dates[0]
            value = bond_yield_hist.loc[next_date, isin]
        else:
            value = None
    return value

def simple_auction_strat(auction_hist, bond_yield_hist, duration_invested, benchmark, term_strat):
    '''
    On the Friday preceding the auction, we sell the bonds that have been announced (in a benchmarked ptf, we 'don't
    buy') and buy the bond next to it, in order not to have duration exposure. On the day of the auction, we buy
    the auction bond back and sell the other bond
    '''
    if term_strat == 'long':
        auction_hist = auction_hist[auction_hist['type of auction'] == 'adju_LT']
        # We only consider long end buckets for our long end strategy
        auction_hist = auction_hist[~auction_hist['maturity bucket'].isin(['0-3', '3-5', '5-7'])]

    # We remove problematic isins
    problematic_isins = ['FR0010466938', 'FR0011317783', 'FR0011486067', 'FR0013451507', 'FR0013516549', 'FR0014002WK3', 'FR0014002JM6', 'FR0014007L00', 'FR0014009O62']
    auction_hist = auction_hist[~auction_hist['ISIN code'].isin(problematic_isins)]

    strat_df = auction_hist.copy()
    strat_df = strat_df[['auction date', 'ISIN code', 'line', 'outstanding ratio', 'relative price', 'maturity bucket']].copy()
    # We create a calendar of implementation days taking the auction dates in auction_hist and shifting back to the Friday previous
    strat_df['previous_friday'] = strat_df['auction date'].apply(
        lambda x: x - pd.Timedelta(days=x.weekday())
    ) - BDay(1)

    start_date = '2015-01-01'
    testing_dates = strat_df[strat_df['auction date'] >= start_date]
    PnL = {}
    for implementation_date in testing_dates['previous_friday'].unique():
        # if implementation_date == pd.Timestamp('2015-05-04 00:00:00'):
        #     breakpoint()
        PnL[implementation_date] = {}
        # We take the isins that have been announced at auction
        isins = strat_df[strat_df['previous_friday'] == implementation_date]

        trade_duration = 4
        for index, row in isins.iterrows():
            isin = row['ISIN code']
            maturity_bucket = row['maturity bucket']
            maturity_bucket_row = bond_yield_hist.loc['maturity bucket']
            matching_isins = maturity_bucket_row[maturity_bucket_row == maturity_bucket].index
            # Trade
            open_yield_short = extract_yield(bond_yield_hist, implementation_date, isin)
            close_yield_short = extract_yield(bond_yield_hist, implementation_date + BDay(trade_duration), isin)
            open_yield_long = extract_yield(bond_yield_hist, implementation_date, matching_isins[0])
            close_yield_long = extract_yield(bond_yield_hist, implementation_date + BDay(trade_duration),
                                             matching_isins[0])
            curve_open = open_yield_short - open_yield_long
            curve_close = close_yield_short - close_yield_long
            ret = (curve_close - curve_open) * duration_invested
            PnL[implementation_date][isin] = ret

    pnl_plotter(PnL)

def future_strat(auction_hist, bond_yield_hist, future_data, duration_invested, benchmark, weight, term_strat):
    '''
    On the Friday preceding the auction, we sell the 10 year bond that has been announced and buy 10 year future,in
    order not to have duration exposure. On the day of the auction, we buy
    the auction bond back and sell the 10 year future
    '''
    if term_strat == 'long':
        auction_hist = auction_hist[auction_hist['type of auction'] == 'adju_LT']
        # We only consider long end buckets for our long end strategy
        auction_hist = auction_hist[~auction_hist['maturity bucket'].isin(['0-3', '3-5', '5-7'])]

    # We remove problematic isins
    problematic_isins = ['FR0010466938', 'FR0011317783', 'FR0011486067', 'FR0013451507', 'FR0013516549',
                         'FR0014002WK3', 'FR0014002JM6', 'FR0014007L00', 'FR0014009O62']
    auction_hist = auction_hist[~auction_hist['ISIN code'].isin(problematic_isins)]

    strat_df = auction_hist.copy()
    # We ony take the 10 year bucket
    strat_df = strat_df[strat_df['maturity bucket'] == '7-10']
    strat_df = strat_df[['auction date', 'ISIN code', 'line', 'outstanding ratio', 'relative price', 'maturity bucket']].copy()
    strat_df['previous_friday'] = strat_df['auction date'].apply(
            lambda x: x - pd.Timedelta(days=x.weekday())
        ) - BDay(1)

    future_data = future_data.set_index('date')

    start_date = '2015-01-01'
    testing_dates = strat_df[strat_df['auction date'] >= start_date]
    PnL = {}
    for implementation_date in testing_dates['previous_friday'].unique():
        PnL[implementation_date] = {}
        # We take the isins that have been announced at auction
        isins = strat_df[strat_df['previous_friday'] == implementation_date]

        trade_duration = 4
        for index, row in isins.iterrows():
            isin = row['ISIN code']
            # Trade
            open_yield_short = extract_yield(bond_yield_hist, implementation_date, isin)
            close_yield_short = extract_yield(bond_yield_hist, implementation_date + BDay(trade_duration), isin)
            open_future_long = extract_yield(future_data, implementation_date, 'OAT Comdty')
            close_future_long = extract_yield(future_data, implementation_date + BDay(trade_duration), 'OAT Comdty')
            ret_bond = (close_yield_short - open_yield_short) * duration_invested
            ret_future = ((close_future_long / open_future_long)-1 ) * 100 * weight
            ret = ret_future + ret_bond
            PnL[implementation_date][isin] = ret

    pnl_plotter(PnL)

def auction_strat(auction_hist, bond_yield_hist, duration_invested):
    '''
    On the Friday preceding an auction, AFT announces the bonds to be auctioned on the following
    Thursday and the upper and lower limits of the total amount to be auctioned.
    Auctions of long- term OATs (maturities of eight and a half years or more in 2022)
    are held on the first Thursday of the month at 10:50am,

    The long term bond strategy will involve calculating the probability that a certain long end
    bond is auctioned in the coming week, selling it on the preceding Monday and buying it back
    on the following Friday.

    We will call the Monday the implementation day and the Friday the liquidation day.
    Basically, we calculate a calendar of implementation days. For each implementation day, for
    each long end bond present in the auction history list that came before, we calculate the likelyhood that it
    will be tapped
    '''

    auction_hist = auction_hist[auction_hist['type of auction'] == 'adju_LT']
    strat_df = auction_hist[['auction date', 'ISIN code', 'line', 'outstanding ratio', 'relative price', 'maturity bucket']]
    # We only consider long end buckets for our long end strategy
    strat_df = strat_df[~strat_df['maturity bucket'].isin(['0-3', '3-5', '5-7'])]
    strat_df['signal_difference'] = strat_df['relative price'] - strat_df['outstanding ratio']

    # We create a calendar of implementation days taking the auction dates in auction_hist and shifting back
    # to the monday previous
    strat_df['previous_monday'] = strat_df['auction date'].apply(
        lambda x: x - pd.Timedelta(days=x.weekday())
    ) - BDay(5)

    start_date = '2015-01-01'
    testing_dates = strat_df[strat_df['auction date'] >= start_date]
    PnL = {}
    for implementation_date in testing_dates['previous_monday'].unique():
        PnL[implementation_date] = {}

        start_test_date = implementation_date - pd.Timedelta(days=720)
        temp_df = strat_df[(strat_df['auction date'] <= implementation_date) & (strat_df['auction date'] >= start_test_date)]
        filtered_df = temp_df.dropna(subset=['relative price', 'outstanding ratio'])
        # We calculate the highest signal difference, i.e. the likeliest bond to auction,  for each bucket
        idx = filtered_df.groupby('maturity bucket')['signal_difference'].agg(lambda x: x.idxmax() if x.max() == x.max() else None)
        valid_idx = idx.dropna()
        highest_signal_diff_isins = filtered_df.loc[valid_idx, ['maturity bucket', 'ISIN code', 'signal_difference']]
        # Amont the highest_signal_diff_isins, let's only keep the highest 3
        highest_signal_diff_isins = highest_signal_diff_isins.sort_values('signal_difference', ascending=False).reset_index()
        remaining_isins = highest_signal_diff_isins.iloc[3:]
        highest_signal_diff_isins = highest_signal_diff_isins.head(3)

        # We remove the isins that are not present in the bond yield hist
        highest_signal_diff_isins = highest_signal_diff_isins[highest_signal_diff_isins['ISIN code'].isin(list(bond_yield_hist.columns))]

        # We update the bond_yield_hist maturity buckets according to the implementation date
        bond_yield_hist = update_bond_hist_maturity_bucket(bond_yield_hist, implementation_date)

        # We now implement the trading strategy
        trade_duration = 4
        for index, row in highest_signal_diff_isins.iterrows():
            isin = row['ISIN code']
            maturity_bucket = row['maturity bucket']
            maturity_bucket_row = bond_yield_hist.loc['maturity bucket']
            matching_isins = maturity_bucket_row[maturity_bucket_row == maturity_bucket].index
            # Trade
            open_yield_short = extract_yield(bond_yield_hist, implementation_date, isin)
            close_yield_short = extract_yield(bond_yield_hist, implementation_date + BDay(trade_duration), isin)
            # check if isin is in matching_isins, if so remove
            matching_isins = matching_isins[matching_isins != isin]
            if len(matching_isins) == 1:
                #ret = 0
                open_yield_long = extract_yield(bond_yield_hist, implementation_date, matching_isins[0])
                close_yield_long = extract_yield(bond_yield_hist, implementation_date + BDay(trade_duration), matching_isins[0])
                curve_open = open_yield_short - open_yield_long
                curve_close = close_yield_short - close_yield_long
                ret = curve_close - curve_open
            else:
                matching_isins = matching_isins[:2]
                k = 2
                matching_isins = matching_isins.sort_values()
                open_yield_long1 = extract_yield(bond_yield_hist, implementation_date, matching_isins[0])
                open_yield_long2 = extract_yield(bond_yield_hist, implementation_date, matching_isins[1])
                close_yield_long1 = extract_yield(bond_yield_hist, implementation_date + BDay(trade_duration), matching_isins[0])
                close_yield_long2 = extract_yield(bond_yield_hist, implementation_date + BDay(trade_duration), matching_isins[1])
                # We here manage the case in which one of the wings is not present in the bond yield hist
                if np.isnan(open_yield_long1) or np.isnan(open_yield_long2):
                    k = 1
                    if np.isnan(open_yield_long1):
                        open_yield_long1 = 0
                        close_yield_long1 = 0
                    else:
                        open_yield_long2 = 0
                        close_yield_long2 = 0
                fly_open = open_yield_short * k - open_yield_long1 - open_yield_long2
                fly_close = close_yield_short * k - close_yield_long1 - close_yield_long2
                ret = fly_close - fly_open
            # Append ret to PnL
            PnL[implementation_date][isin] = ret * duration_invested

    pnl_plotter(PnL)

def pnl_plotter(PnL):
    PnLSum = 0
    PnLs = []
    for i in PnL:
        for j in PnL[i]:
            pnl = PnL[i][j]
            if np.isnan(pnl):
                pnl = 0
            PnLSum += pnl
            PnLs.append(pnl)

    data = []
    for date, bonds in PnL.items():
        for bond, pnl in bonds.items():
            data.append({'implementation_date': date, 'bond': bond, 'PnL': pnl})

    pnl_df = pd.DataFrame(data)
    pnl_df['cumulative_PnL'] = pnl_df['PnL'].cumsum()
    pnl_df = pnl_df.ffill()
    # Plot PnL
    chart = pnl_df.plot(x='implementation_date', y='cumulative_PnL', kind='line', linewidth=1, linestyle='-', color='blue')
    chart.set_xlabel('Date')
    chart.set_ylabel('Performance')
    plt.show()


def main():
    # Future data is imported for benchmark creation
    # Bond data is imported for  active bond list
    # Historical bond price data is imported in order to study historical cash prices
    # Historical yield data is imported in order to have historical yields if need be
    # Auction data is imported in order to study past auctions
    future_data = pd.read_excel('futures_dataset.xlsx')
    bond_data = pd.read_excel('bond_data.xlsx')
    bond_yield_hist = pd.read_excel('bond_hist_data.xlsx', sheet_name='yield_data')
    bond_price_hist = pd.read_excel('bond_hist_data.xlsx', sheet_name='price_data')
    auction_hist = pd.read_excel('auction_hist.xlsx')
    #synd_hist = pd.read_excel('syndication_hist.xlsx')


    # Data cleaning
    future_data = clean_future_data(future_data)
    bond_yield_hist = clean_bond_hist(bond_yield_hist, bond_data)
    bond_price_hist = clean_bond_price_hist(bond_price_hist)
    auction_hist = clean_auction_hist(auction_hist)

    # Benchmark creation for French Portfolio of 20bn
    # We assume 20% of the portfolio is invested in France, i.e. 4bn
    # We assume the benchmark is made up of 250m of GTFRF2Y, 950m of GTFRF5Y, 1.65bn of GTFRF10Y and 1.15bn of GTFRF30Y
    # In order to come to a weighted average duration of that of the French debt (8.5 years) according to
    # https://www.aft.gouv.fr/en/node/452
    # the relative weights of the 4 benchmark bonds in the portfolios are 1.3%, 4.8%, 8.3% and 5.8% respectively
    # The relative duration, by multiplying the bond duration by the weights, is 2.46 years
    # We have approximate durations, for the 4 buckets, of 0.02, 0.2, 0.74 and 1.5
    # We can now calculate the returns of the benchmark bonds by multiplying the relative duration by the change in yield

    benchmark = benchmark_creation(future_data)

    #auction_des_stats(auction_hist)
    stats_format, bond_price_hist = study_bond_price_hist(bond_price_hist, bond_data)

    # We now bring our two signals together, i.e. the issuance signal and the price signal
    # We do this by merging the two dataframes on the ISIN and the date
    stats_format = stats_format[['ISIN', 'date', 'relative price']]
    stats_format.rename(columns={'ISIN': 'ISIN code', 'date': 'auction date'}, inplace=True)
    auction_hist = auction_hist.merge(stats_format, on=['ISIN code', 'auction date'], how='left')

    # I assume, as per above, that my 10 year benchmark has 0.74 of duration
    duration_invested = 0.74
    weight_10_y = 0.083

    simple_auction_strat(auction_hist, bond_yield_hist, duration_invested, benchmark, term_strat='long')
    future_strat(auction_hist, bond_yield_hist, future_data, duration_invested, benchmark, weight_10_y, term_strat='long')
    auction_strat(auction_hist, bond_yield_hist, duration_invested)

    breakpoint()


main()
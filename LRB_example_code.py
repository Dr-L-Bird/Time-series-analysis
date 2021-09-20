import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing as mp

"""
Runs multiple ARIMA modes at the same time
Version control handled via Github desktop app
Program run starts at __main__ runs def model_section multiple times 
to create trade figures.
Processing 2000 - 2020 trade data for mineral commodities 
Output figures with 2020 forecasted, forecast error and data for 2020
def model_section depends on def divide_list, def is_zero, def virus_for_figure 
these can be moved to new files for re-use 
Windows / Mac compatibility:
Will run on both with path changes at lines 200 and 213 
currently in Mac compatibility
"""


def divide_list(list_values):
    '''
    :param list_values: long list that can't be run all at once, can handle blocks of 4,6,8 or 12.
    more results in fatale error
    :return: list divided into blocks to be handled individually by each core
    Note: Adjust block_size based on physical processor core count e.g. intel core i3 set to 4
    '''
    block_size = 4
    for i in range(0, len(list_values), block_size):
        yield list_values[i:i + block_size]


def is_zero(x):
    """
    :param x: value need to check to see if negative, trade in figures
    can't be negative
    :return: 0 or original value, value can not be less than 0 for plot
    """
    if x <= 0:
        x = 1
    else:
        x = x
    return x


def virus_for_figure(viruses, text):
    '''
    From the viruses data frame need date of first appearance and length of time
    :param viruses:
    :param text:
    :return:
    '''
    sars = viruses[text]
    sars = sars.dropna(how='all', axis=0)
    sars_values = sars.index.tolist()
    start = min(sars_values)
    length = max(sars_values) - min(sars_values)

    return start, length


def model_section(commodity, df, viruses):

    # Getting time periods for virus and economic events stored in dataframe
    # this could be done outside of def and passed to multiprosses
    # current overhead low
    sars_start, sars_length = virus_for_figure(viruses, 'SARS')
    h_start, h_length = virus_for_figure(viruses, 'H1N1/09')
    covid_start, convid_length = virus_for_figure(viruses, 'COVID-19')
    gr_start, gr_length = virus_for_figure(viruses, 'Great Recession')
    t_w_start, t_w_length = virus_for_figure(viruses, 'US China tariffs')

    # get specific commodity from data frame
    df_n = df[commodity].copy().to_frame()
    df_n = df_n.astype(float)
    df_n.columns = [commodity]
    first_non_zero = (df_n[commodity] > 5).idxmax(axis=1, skipna=True)
    df_n = df_n.loc[first_non_zero:]

    # want to plot data that has 2020 so creating a copy for the plots
    df_original = df_n.copy()

    # cutting out any 2020 data as want to forecast 2020
    df_n = df_n.loc[:'2019-12-01']
    # test and train dates
    train_date = '2018-12-01'
    test_date = '2018-12-01'

    # Smooth data to remove noise
    span = 12
    alpha = 2 / (span + 1)
    df_n[commodity] = df_n[commodity].ewm(alpha=alpha, adjust=False).mean()
    df_original[commodity] = df_original[commodity].ewm(alpha=alpha, adjust=False).mean()

    '''
    auto_arima function seeks to identify the most optimal 
    parameters for an ARIMA model, and returns a fitted ARIMA model.     
    use_sarimax is used later for the program to decide if seasonality is a 
    component of the projections. 0 no seasonality will use arima else will use
    seasonal arima model
    '''

    aut = auto_arima(df_n, stepwise=False, seasonal=True, m=12)
    params = aut.to_dict()
    use_sarimax = sum(params['seasonal_order'][0:3])

    if use_sarimax == 0:
        train = df_n[:train_date]
        test = df_n[test_date:]
        # test projections
        model_arima = ARIMA(train[commodity], order=params['order'])
        results_arima = model_arima.fit()
        start = len(train)
        end = len(train) + len(test) - 1
        predictions = results_arima.predict(start=start, end=end, dynamic=False, typ='levels').rename('Predictions')
        # forecast projections
        model = ARIMA(df_n[commodity], order=params['order'])
        results = model.fit()
        fcast = results.predict(len(df_n), len(df_n) + 12, typ='levels').rename('Forecast')
        # Forecast confidence levels set to 90%
        fcast_90_conf = results.get_prediction(len(df_n), len(df_n) + 12, typ='levels').summary_frame(alpha=0.10)
    else:
        # Data has a unit root and is non-stationary
        #  using auto_arima with seasonality set to True
        train = df_n[:train_date]
        test = df_n[test_date:]
        # test projections
        model_sArima = SARIMAX(train[commodity], order=params['order'], seasonal_order=params['seasonal_order'])
        results_p = model_sArima.fit()
        start = len(train)
        end = len(train) + len(test) - 1
        predictions = results_p.predict(start=start, end=end, dynamic=False, typ='levels').rename('Predictions')
        # forecast projections
        model = SARIMAX(df_n[commodity], order=params['order'], seasonal_order=params['seasonal_order'])
        results = model.fit()
        fcast = results.predict(len(df_n), len(df_n) + 12, typ='levels').rename('Forecast')
        # Forecast confidence levels set to 90%
        fcast_90_conf = results.get_prediction(len(df_n), len(df_n) + 12, typ='levels').summary_frame(alpha=0.10)

    box_one = df_original[commodity].max()
    fcast_90_conf['mean_ci_lower'] = fcast_90_conf['mean_ci_lower'].apply(is_zero)
    box_two = fcast_90_conf['mean_ci_upper'].max()
    box_m = max(box_one, box_two)

    '''
    Create figures
    These will be saved in the root folder
    '''
    plt.figure()
    title = str(commodity) + '(China)'
    ylabel = 'Mg'
    xlabel = 'Date'
    # figure data
    ax = df_original[commodity].plot(legend=True, figsize=(12, 6), title=title)
    # plot future projections
    fcast.plot(legend=True, color='r')
    '''
    Can plot predictions to assess model but already plotting 2020 which fcast 
    aimed to prediction. Currently displaying confidence levels to assess error
    '''
    # predictions.plot(legend=True)
    ax.fill_between(fcast_90_conf.index, fcast_90_conf['mean_ci_lower'], fcast_90_conf['mean_ci_upper'], color='k', alpha=0.4)
    # figure annotations
    plt.text('2003-03-01', box_m, 'Sars', fontsize=10) # use dates to move text postitons forward or back, box_m is height positon
    plt.text('2009-07-01', box_m, 'H1N1/09', fontsize=10)
    plt.text(covid_start, box_m, 'Covid-19', fontsize=10)
    plt.text('2007-09-01', box_m, 'Recession', fontsize=10)
    plt.text('2017-06-01', box_m, 'US china tariffs', fontsize=10)
    p_sars = patches.Rectangle((sars_start, 0), sars_length, box_m, facecolor="silver", alpha=0.3)
    p_H1N1 = patches.Rectangle((h_start, 0), h_length, box_m, facecolor="silver", alpha=0.3)
    p_covid = patches.Rectangle((covid_start, 0), convid_length, box_m, facecolor="lavender", alpha=0.3)
    p_great_reces = patches.Rectangle((gr_start, 0),  gr_length, box_m, facecolor="lightskyblue", alpha=0.3)
    p_trade_war = patches.Rectangle((t_w_start, 0), t_w_length, box_m, facecolor="lightskyblue", alpha=0.3)
    # add patches to figure
    ax.add_patch(p_sars)
    ax.add_patch(p_H1N1)
    ax.add_patch(p_covid)
    ax.add_patch(p_great_reces)
    ax.add_patch(p_trade_war)
    # figure legend and clean up
    plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2)
    ax.autoscale(axis='x', tight=True)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    # save and close
    plt.savefig(commodity + " China  " + ".png")
    plt.close()


if __name__ == '__main__':

    '''
    Read in trade data to pandas data frame and clean up 
    before passing variables to target def for multiprocessing run
    Multiple Arima models will be run at the same time across processor cores
    '''

    read_in = pd.read_excel('~/OneDrive/Python/TimeSeries/China_May/China_May_test.xlsx', sheet_name='commodities')
    # List of commodities in Data column due to how dataframe was saved
    comod_list = list(read_in['Date'])
    # results of previous data extract need to be transposed
    # and certain column need to be dropped
    read_in = read_in.drop(columns=['Commodity', 'Direction', 'trend', 'limiter']).T
    read_in.columns = read_in.iloc[0]
    read_in = read_in.drop(read_in.index[0])
    read_in = read_in.rename_axis("Date")

    # if testing figure generation only want to create one figure
    # to test figure design uncomment 212
    # comod_list = ['Zinc Smelter Imports']

    # index need to be set as datetime for time series analysis
    read_in.index = pd.to_datetime(read_in.index)
    read_in.index.freq = "MS"

    # read in additional data on dates of viruses and economic events
    viruses = pd.read_excel('virus.xlsx', parse_dates=['date'], index_col='date')

    # break up codes in to batches of 6 so not to overload cpu core
    # changed to 8 as FX-8350e has 8 physical cores
    list_to_run = list(divide_list(comod_list))
    for sub_list in list_to_run:
        jobs = []
        for item in sub_list:
            multi_process = mp.Process(target=model_section, args=(item, read_in, viruses))
            jobs.append(multi_process)
            multi_process.start()

        for j in jobs:
            # make sure processes are completed before main process
            j.join()
#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:10:03 2022

@author: alice
"""


# import necessary modules

import matplotlib.pyplot as plt
import pandas as pd
import datetime 
import sys
import requests
import numpy as np
from datetime import timedelta

# FOR DEVICE 1
#------------------------------------------------------------------
# DEFINE FUNCTIONS TO GENERATE TEMPLATE CI AND PRICE FORECASTS

# generate a dataframe of times and CI values 
# parameters: length --> length of dataframes required, usually 48

def make_ci(length):
    
    saved_ci = pd.read_csv(r'...\Testing forecasts\Plug forecasts\CI_1603.csv', header=None)
    ci_df = pd.DataFrame(saved_ci)
    
    return ci_df
        
def make_price(length):
    
    saved_prices = pd.read_csv(r'...\Testing forecasts\Plug forecasts\prices_1603.csv', header=None)
    prices_df = pd.DataFrame(saved_prices)
    
    return prices_df
        
#-------------------------------------------------------------------------        
# DEFINE FUNCTION TO GENERATE TEMPLATE DEVICE INFORMATION DATAFRAME        
        
# generate a dataframe of device information

def make_device_info():
    # Create fake data frames for dev
    column_names = ['device_name','power','on_period','use_OFF','s_off','e_off','use_deadline','deadline']
    
    # Here you can input the details you would like to include for devices
    # include data in the following formats
     
    #  | device_name | power   | on_period | use_OFF    | s_off | e_off | use_deadline | deadline |
    #  | string      | integer | integer   | true/false | HH:MM | HH:MM | true/false   | HH:MM    |
    
    
    d1_values = ['P2','8','4','false','09:30','10:00','true','10:00']
    d1_data_raw = [column_names, d1_values]
    d1_df_raw = pd.DataFrame(d1_data_raw)
    d1_df = d1_df_raw.transpose()

    device_data_all = [d1_df.iloc[:,0],d1_df.iloc[:,1]]
    device_info_df = pd.DataFrame(device_data_all).transpose()
    device_info_df.columns = [0,1]
        
    return device_info_df    
        
#------------------------------------------------------------------------------------------
# DEFINE FUNCTION TO GET REAL CI DATA FROM API

def get_ci_data():
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    co2_base_url = "https://api.carbonintensity.org.uk"
    co2_url = (co2_base_url+"/intensity/"+now+"/fw24h")
    co2 = requests.get(co2_url)
    
    intensity = co2.json()
    
    ci_df = pd.json_normalize(intensity, 'data')
    ci_df = ci_df[['from','intensity.forecast']]
    
    ci_df.iloc[:,0] = [w[11:16] for w in ci_df.iloc[:,0]]
    ci_df = ci_df.iloc[1:,:]
    ci_df = ci_df.reset_index(drop=True)
    
    return ci_df

def get_price_data():
    
    now = datetime.datetime.today()
    now_fw24 = datetime.datetime.today() + datetime.timedelta(days=1)
    
    last_year = now + timedelta(days=-365)
    last_year_fw24 = now_fw24 + timedelta(days=-365)
    
    now_str = last_year.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_fw24_str = last_year_fw24.strftime("%Y-%m-%dT%H:%M:%SZ")

    #now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    #now_fw24_str = now_fw24.strftime("%Y-%m-%dT%H:%M:%SZ")

    url = ('https://api.octopus.energy/v1/products/AGILE-18-02-21/' + 
             'electricity-tariffs/E-1R-AGILE-18-02-21-C/standard-unit-rates/' + 
             '?period_from=' + now_str +'&period_to=' + now_fw24_str)
    r = requests.get(url)
    output_dict = r.json()

    valid_from = [x['valid_from'] for x in output_dict['results']]
    value_exc_vat = [x['value_exc_vat'] for x in output_dict['results']]
    
    prices_data = [valid_from, value_exc_vat]

    prices_df_raw = pd.DataFrame(prices_data)
    
    prices_df = prices_df_raw.transpose()
    
    prices_df.iloc[:,0] = [w[11:16] for w in prices_df.iloc[:,0]]
    
    prices_df = prices_df.iloc[:-1,:]
    
    prices_reset = prices_df.reindex(index=prices_df.index[::-1])
    prices_df = prices_reset.reset_index(drop=True)
    
    return prices_df
    
#----------------------------------------------------------------------------------
# DEFINE FUNCTION TO GENERATE DEVICE INFORMATION DATAFRAME FROM NODE RED

def get_device_info():
    
    device_info = pd.read_csv(r'...\device_1_info.csv', header=None)
    device_df = pd.DataFrame(device_info)
    
    start_OFF_dtime = pd.Timestamp(device_df.iloc[4,1], unit='ms')
    end_OFF_dtime = pd.Timestamp(device_df.iloc[5,1], unit='ms')
    
    add_hour = datetime.timedelta(hours = 0)
    
    start_OFF_corrected = str(start_OFF_dtime + add_hour)
    end_OFF_corrected = str(end_OFF_dtime + add_hour)
    
    device_df.iloc[4,1] = start_OFF_corrected[11:16]
    device_df.iloc[5,1] = end_OFF_corrected[11:16]
    
    deadline_raw = device_df.iloc[7,1]
    deadline_long = str(pd.Timestamp(deadline_raw, unit='ms'))
    device_df.iloc[7,1] = deadline_long[11:16]
    
    return device_df
    
#------------------------------------------------------------------------      
# DEFINE FUNCTION TO MAKE PRICE WEIGHTING

def make_pr_weight(weight):
    price_weight = weight
    return price_weight

#------------------------------------------------------------------------
# DEFINE FUNCTION TO GET PRICE WEIGHTING FROM NODE RED    

def get_pr_weight():
    price_weight = pd.read_csv(r'...\price_weighting_1.csv', header=None)
    price_weight = price_weight[0][0]
    return price_weight

#--------------------------------------------------------------------------
# DEFINE FUNCTION FOR PLOTS

def plot_data(figure, x_data, y_data, x_label, y_label, title):
    plt.figure(figure)
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    
    freq_x = 4
    plt.xticks(np.arange(0, 48, freq_x))
  
    plt.title(title)
    
#--------------------------------------------------------------------------

# CHOOSE BRANCH OF CODE TEST/REAL DATA

# comment/uncomment one of the following to run code with test or real data
forecast_branch = 'test'
#forecast_branch = 'real'

#device_branch = 'test'
device_branch = 'real'

if forecast_branch == 'test':
    # edit these values to suit debugging requirements
    ci_df = make_ci(48)
    pr_df = make_price(48)

    
elif forecast_branch == 'real':
    ci_df = get_ci_data()
    pr_df = get_price_data()
    
    
if device_branch == 'test':
    # edit these values to suit debugging requirements
    device_info_df = make_device_info()
    price_weight = make_pr_weight(0.5)
    
elif device_branch == 'real':
    device_info_df = get_device_info()
    price_weight = get_pr_weight()
 
#---------------------------------------------------------------------------
# RETRIEVE DEVICE VARIABLES FROM DATAFRAME

device_name = device_info_df.iloc[0,1]
power = int(device_info_df.iloc[1,1])
on_period = int(device_info_df.iloc[2,1])
use_OFF = device_info_df.iloc[3,1]
s_off = device_info_df.iloc[4,1]
e_off = device_info_df.iloc[5,1]
use_deadline = device_info_df.iloc[6,1]
deadline = device_info_df.iloc[7,1]


#---------------------------------------------------------------------------
# NORMALISE DATA

# make CI and Price dataframes the same length
time_length = min(len(ci_df), len(pr_df))
ci_df_crop = ci_df.iloc[:time_length, :]
pr_df_crop = pr_df.iloc[:time_length,:]

# normalise values
norm_ci = ci_df_crop.iloc[:,1]/ci_df_crop.iloc[:,1].abs().max()
norm_pr = pr_df_crop.iloc[:,1]/pr_df_crop.iloc[:,1].abs().max()

# replace into CI and Price cropped dfs
ci_df_crop.iloc[:,1] = norm_ci
pr_df_crop.iloc[:,1] = norm_pr

# plot CI and Price --> comment out if not needed
plot_data(0,ci_df_crop.iloc[:,0], ci_df_crop.iloc[:,1], 'times', 'CI', 'Normalised Forecast Values')
plot_data(0,pr_df_crop.iloc[:,0], pr_df_crop.iloc[:,1], 'times', 'Price', 'Normalised Forecast Values')

#----------------------------------------------------------------------------------------------------
# APPLY WEIGHTINGS AND CREATE COST VARIABLE DF

# get price and ci weightings
p = price_weight
c = 1 - price_weight

# make new dataframe for comparison
compare_df = pd.DataFrame(zip(ci_df_crop.iloc[:,0], ci_df_crop.iloc[:,1], pr_df_crop.iloc[:,1]), columns=["Time", "CI", "Price"])

# apply optimisation 
z1 = c*compare_df['CI'] # 1st column is CI
z2 = p*compare_df['Price'] # 2nd column is Price

Z = z1.add(z2)

Z_data = [compare_df["Time"], Z]
Z_df = pd.DataFrame(Z_data).transpose()

plt.figure(3)
plt.plot(Z_df.iloc[:,0], Z_df.iloc[:,1], label='Z cost')
plt.plot(compare_df.iloc[:,0], compare_df.iloc[:,1], label='norm CI')
plt.plot(compare_df.iloc[:,0], compare_df.iloc[:,2], label='norm Price')
plt.xticks(ticks=compare_df.iloc[:,0][0::2], labels=compare_df.iloc[:,0][0::2], rotation=90)
plt.title('Normalised Forecasts 16-03, price weighting = 0.5')
plt.legend(loc='upper right')


#----------------------------------------------------------
# DEFINE THE DIFFERENT OPTIMISATION FUNCTIONS

# OP1 --> use_OFF == false && use_deadline == false
# OP2 --> use_OFF == true && use_deadline == false
# OP3 --> use_OFF == false && use_deadline == true
# OP4 --> use_OFF == true && use_deadline == true

#------------------------------------------------------------------------
# OP1

def use_op1():
    
    a = Z_df.index[:-int(on_period)].values
    counter = 0
    sum_results = []

    for each in a:
        # create list of values to sum
        position = a[counter]
        to_sum = Z_df.iloc[int(position):int(position+on_period), 1]
        #sum each possible scenario and save to results vector
        sum_results.append(sum(to_sum))
        counter = counter + 1

    results = [Z_df.iloc[:-int(on_period),0].values, sum_results]
    results_df = pd.DataFrame(results).transpose()                                  

    # find minimum time                
    y_min = results_df[1].min()
    min_time_index = results_df.index[results_df.iloc[:,1] == y_min]
    min_time = results_df.iloc[min_time_index,0]
    
    if min_time.size > 1:
        min_time = min_time.iloc[0]
        start_time_op2 = min_time
    else:
        start_time_op2 = min_time.to_string(index=False)

    # find time to turn device off
    tstamp = pd.Timestamp(start_time_op2)
    hours_toadd = datetime.timedelta(hours = on_period*0.5)

    end_time_full = str(tstamp + hours_toadd)
    end_time = end_time_full[11:16]

    # Print time at which Z is minimum
    print('use op1')
    print('Start device at: ' + start_time_op2)
    print('End device at: ' + end_time)

    # PRINT RESULTS TO CSV              
    csv_data = {'Device': [device_name],'Start Time': [start_time_op2], 'End Time':[end_time]}
    csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
    csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)

#------------------------------------------------------------------------
# OP2

def use_op2():

    # get positional information: what index does no_device_start have in the 
    # normalised ci/pr lists?
    no_st_time_idx = ci_df_crop.index[ci_df_crop.iloc[:,0] == s_off].tolist()
    no_end_time_idx = ci_df_crop.index[ci_df_crop.iloc[:,0] == e_off].tolist()

    no_st_time_str = str(no_st_time_idx)
    no_end_time_str = str(no_end_time_idx)
    
    # make adjustment for if no st/end time in forecast list
    if no_st_time_str == '[]':
        no_st_time_str = str('[' + str(pr_df_crop.index[0]) + ']')
    if no_end_time_str == '[]':
        no_end_time_str = str('[' + str(pr_df_crop.index[0]) + ']')

    # else split as usual and use 2 split run

    if int(no_st_time_str[1:-1]) < on_period:
        
        # not possible to make 1st split. Only consider second split.
        # make cut at e_off
        cut2_pr = pr_df_crop.iloc[int(no_end_time_str[1:-1]):,:]
        cut2_ci = ci_df_crop.iloc[int(no_end_time_str[1:-1]):,:]
        
        compare_df_2 = pd.DataFrame(zip(cut2_pr.iloc[:,0], cut2_pr.iloc[:,1], cut2_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
        
        # apply optimisation using only CUT 2
        z1_2 = p*compare_df_2["Price"] # 1st column is price
        z2_2 = c*compare_df_2["CI"] # 2nd column is CI
        
        Z_2 = z1_2.add(z2_2)
        
        # Run optimisation for first split
        Z_data_2 = [compare_df_2["Time"], Z_2]
        Z_df_2 = pd.DataFrame(Z_data_2).transpose()
        
        a_2 = Z_df_2.index[:-int(on_period)].values
        counter_2 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_2) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_2:
            # create list of values to sum
            position = a_2[counter_2]
            to_sum = Z_df_2.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_2 = counter_2 + 1
        
        results_2 = [Z_df_2.iloc[:-int(on_period),0].values, sum_results]
        results_df_2 = pd.DataFrame(results_2).transpose()
        
        # find minimum time               
        y_min_2 = results_df_2[1].min()
        min_time_index_2 = results_df_2.index[results_df_2.iloc[:,1] == y_min_2]
        min_time_2 = results_df_2.iloc[min_time_index_2,0]
        
        if min_time_2.size > 1:
            min_time = min_time_2.iloc[0]
            start_time_2 = min_time
        else:
            start_time_2 = min_time_2.to_string(index=False)
        
        
        # find time to turn device off
        tstamp = pd.Timestamp(start_time_2)
        hours_toadd = datetime.timedelta(hours = on_period*0.5)
        
        end_time_full = str(tstamp + hours_toadd)
        end_time = end_time_full[11:16]
        
        
        # Print time at which Z is minimum
        print('use op2')
        print('use cut 2')
        print('Start device at: ' + start_time_2)
        print('End device at: ' + end_time)
        
        # PRINT RESULTS TO CSV              
        csv_data = {'Device': [device_name],'Start Time': [start_time_2], 'End Time':[end_time]}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
       
    elif int(pr_df_crop.index[-1]) - int(no_end_time_str[1:-1]) < on_period:
        
        # not possible to make 2nd split. Only consider first split.
        # make cut at s_off - on_period
        
        # make first half of ci and pr
        cut1_pr = pr_df_crop.iloc[:int(no_st_time_str[1:-1])-int(on_period),:]
        cut1_ci = ci_df_crop.iloc[:int(no_st_time_str[1:-1])-int(on_period),:]
        
        compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])

        # apply optimisation using only CUT 1
        z1_1 = p*compare_df_1["Price"] # 1st column is price
        z2_1 = c*compare_df_1["CI"] # 2nd column is CI
        
        Z_1 = z1_1.add(z2_1)
        
        # Run optimisation for first split
        Z_data_1 = [compare_df_1["Time"], Z_1]
        Z_df_1 = pd.DataFrame(Z_data_1).transpose()
        
        a_1 = Z_df_1.index.values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_1) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:,0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        # find minimum time               
        y_min_1 = results_df_1[1].min()
        min_time_index_1 = results_df_1.index[results_df_1.iloc[:,1] == y_min_1]
        min_time_1 = results_df_1.iloc[min_time_index_1,0]
        
        if min_time_1.size > 1:
            min_time = min_time_1.iloc[0]
            start_time_1 = min_time
        else:
            start_time_1 = min_time_1.to_string(index=False)
        
        
        # find time to turn device off
        tstamp = pd.Timestamp(start_time_1)
        hours_toadd = datetime.timedelta(hours = on_period*0.5)
        
        end_time_full = str(tstamp + hours_toadd)
        end_time = end_time_full[11:16]
        
        
        # Print time at which Z is minimum
        print('use op2')
        print('use cut 1')
        print('Start device at: ' + start_time_1)
        print('End device at: ' + end_time)
        
        # PRINT RESULTS TO CSV              
        csv_data = {'Device': [device_name],'Start Time': [start_time_1], 'End Time':[end_time]}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
    
    else:
        
        # split into two cuts and continue with optimisation
    
        # make first half of ci and pr
        cut1_pr = pr_df_crop.iloc[:int(no_st_time_str[1:-1])-int(on_period),:]
        cut1_ci = ci_df_crop.iloc[:int(no_st_time_str[1:-1])-int(on_period),:]
    
        # make second half of ci and pr
        cut2_pr = pr_df_crop.iloc[int(no_end_time_str[1:-1]):,:]
        cut2_ci = ci_df_crop.iloc[int(no_end_time_str[1:-1]):,:]
    
        # make 2 new dataframes for comparison
        compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
        compare_df_2 = pd.DataFrame(zip(cut2_pr.iloc[:,0], cut2_pr.iloc[:,1], cut2_ci.iloc[:,1]), columns=["Time", "Price", "CI"])

        # apply optimisation 
        z1_1 = p*compare_df_1["Price"] # 1st column is price
        z2_1 = c*compare_df_1["CI"] # 2nd column is CI
        
        z1_2 = p*compare_df_2["Price"] # 1st column is price
        z2_2= c*compare_df_2["CI"] # 2nd column is CI
        
        Z_1 = z1_1.add(z2_1)
        Z_2 = z1_2.add(z2_2)
        
        # Run optimisation for first split
        Z_data_1 = [compare_df_1["Time"], Z_1]
        Z_df_1 = pd.DataFrame(Z_data_1).transpose()
        
        a_1 = Z_df_1.index.values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_1) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:,0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        # Run optimisation for second split
        Z_data_2 = [compare_df_2["Time"], Z_2]
        Z_df_2 = pd.DataFrame(Z_data_2).transpose()
        
        a_2 = Z_df_2.index[:-int(on_period)].values
        counter_2 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_2) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_2:
            # create list of values to sum
            position = a_2[counter_2]
            to_sum = Z_df_2.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_2 = counter_2 + 1
        
        results_2 = [Z_df_2.iloc[:-int(on_period),0].values, sum_results]
        results_df_2 = pd.DataFrame(results_2).transpose()
        
        # find minimum time               
        y_min_1 = results_df_1[1].min()
        min_time_index_1 = results_df_1.index[results_df_1.iloc[:,1] == y_min_1]
        min_time_1 = results_df_1.iloc[min_time_index_1,0]
        
        if min_time_1.size > 1:
            min_time = min_time_1.iloc[0]
            start_time_1 = min_time
        else:
            start_time_1 = min_time_1.to_string(index=False)
        
        
        y_min_2 = results_df_2[1].min()
        min_time_index_2 = results_df_2.index[results_df_2.iloc[:,1] == y_min_2]
        min_time_2 = results_df_2.iloc[min_time_index_2,0]
        
        start_time_2 = min_time_2.to_string(index=False)
     
        # Get minimum out of 2 splits
        y_min = min(y_min_1, y_min_2)
        
        if y_min == y_min_1:
            start_time_1 = start_time_1
        elif y_min == y_min_2:
            start_time_1 = start_time_2
        
        # find time to turn device off
        tstamp = pd.Timestamp(start_time_1)
        hours_toadd = datetime.timedelta(hours = on_period*0.5)
        
        end_time_full = str(tstamp + hours_toadd)
        end_time = end_time_full[11:16]
        
        
        # Print time at which Z is minimum
        print('use op2')
        print('two split run')
        print('Start device at: ' + start_time_1)
        print('End device at: ' + end_time)
        
        # PRINT RESULTS TO CSV              
        csv_data = {'Device': [device_name],'Start Time': [start_time_1], 'End Time':[end_time]}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
                         
#---------------------------------------------------------------------------------------------------------
# OP3

def use_op3():
    
    # get positional information: what index does no_device_start have in the 
    # normalised ci/pr lists?
    deadline_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == deadline].tolist()
    
    deadline_int = str(deadline_idx)
    
    if deadline_idx == []:
        print('No possible start time for given constraints')
        csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
        sys.exit()
    
    if int(deadline_int[1:-1])-int(on_period) <= 0:
        print('No possible start time for given constraints')
        csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
        sys.exit()

    
    cut1_pr = pr_df_crop.iloc[:int(deadline_int[1:-1]),:]
    cut1_ci = ci_df_crop.iloc[:int(deadline_int[1:-1]),:]


    # make 2 new dataframes for comparison
    compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
    
    # check cut is long enough for on_period
    if len(cut1_pr) <= on_period:
        print('No possible start time for given constraints')
        csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
        quit()
    
    # apply optimisation 
    z1_1 = p*compare_df_1["Price"] # 1st column is price
    z2_1 = c*compare_df_1["CI"] # 2nd column is CI
    
    Z_1 = z1_1.add(z2_1)
    
    # Run optimisation for first split
    Z_data_1 = [compare_df_1["Time"], Z_1]
    Z_df_1 = pd.DataFrame(Z_data_1).transpose()
    
    a_1 = Z_df_1.index[:-int(on_period)+1].values
    counter_1 = 0 # this is for the purposes of the for loop
    sum_results = []
    
    if len(a_1) == 0:
        print('No possible start time for given constraints')
        csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
        sys.exit()
    
    for each in a_1:
        # create list of values to sum
        position = a_1[counter_1]
        to_sum = Z_df_1.iloc[int(position):int(position+on_period), 1]
        #sum each possible scenario and save to results vector
        sum_results.append(sum(to_sum))
        counter_1 = counter_1 + 1
    
    results_1 = [Z_df_1.iloc[:-int(on_period)+1,0].values, sum_results]
    results_df_1 = pd.DataFrame(results_1).transpose()
    
    # find minimum time               
    y_min_1 = results_df_1[1].min()
    min_time_index_1 = results_df_1.index[results_df_1.iloc[:,1] == y_min_1]
    min_time_1 = results_df_1.iloc[min_time_index_1,0]
    
    if min_time_1.size > 1:
        min_time = min_time_1.iloc[0]
        start_time = min_time
    else:
        start_time = min_time_1.to_string(index=False)
    
    
    # find time to turn device off
    tstamp = pd.Timestamp(start_time)
    hours_toadd = datetime.timedelta(hours = on_period*0.5)
    
    end_time_full = str(tstamp + hours_toadd)
    end_time = end_time_full[11:16]
    
    
    # Print time at which Z is minimum
    print('use op3')
    print('Start device at: ' + start_time)
    print('End device at: ' + end_time)
    
    # PRINT RESULTS TO CSV              
    csv_data = {'Device': [device_name],'Start Time': [start_time], 'End Time':[end_time]}
    csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
    csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
                          
#---------------------------------------------------------------------------------------------------
# OP4

def use_op4():
    
    # Get index positions of s_off, e_off and deadline
    deadline_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == deadline].tolist()
    deadline_int = str(deadline_idx)
    
    no_st_time_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == s_off].tolist()
    no_end_time_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == e_off].tolist()

    no_st_time_int = str(no_st_time_idx)
    no_end_time_int = str(no_end_time_idx)
    
    # adjust no_st and no_end if necessary
    if no_st_time_int == '[]':
        no_st_time_int = str('[' + str(pr_df_crop.index[0]) + ']')
    
    if no_end_time_int == '[]':
        no_end_time_int = str('[' + str(pr_df_crop.index[0]) + ']')
        
    # add tags to check whether cuts are long enough (note cut2 doesn't
    # require this as can project end time beyond forecast vector end)
    cut1_okay = True
    
    # is 'cut1' going to be long enough for device to run?
    if int(no_st_time_int[1:-1])-int(on_period) <= 0:
        cut1_okay = False
    
    # Determine where the deadline sits relative to no_st and no_end
    # 1. deadline is before no_st
    # 2. deadline is between no_st and no_end
    # 3. deadline is after no_end
    
    # Option 1
    if int(deadline_int[1:-1]) <= int(no_st_time_int[1:-1]):
        # use deadline_int as deadline
        
        # check new dataframe is long enough for on_period
        if int(deadline_int[1:-1]) <= on_period:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            
            sys.exit()
        
        # trim of ci and pr        
        cut1_pr = pr_df_crop.iloc[:int(deadline_int[1:-1])-int(on_period)+1,:]
        cut1_ci = ci_df_crop.iloc[:int(deadline_int[1:-1])-int(on_period)+1,:]

        # make 2 new dataframes for comparison
        compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
        
        # apply optimisation 
        z1_1 = p*compare_df_1["Price"] # 1st column is price
        z2_1 = c*compare_df_1["CI"] # 2nd column is CI
        
        Z_1 = z1_1.add(z2_1)
        
        # Run optimisation for first split
        Z_data_1 = [compare_df_1["Time"], Z_1]
        Z_df_1 = pd.DataFrame(Z_data_1).transpose()
        
        a_1 = Z_df_1.index.values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_1) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:,0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        # find minimum time               
        y_min_1 = results_df_1[1].min()
        min_time_index_1 = results_df_1.index[results_df_1.iloc[:,1] == y_min_1]
        min_time_1 = results_df_1.iloc[min_time_index_1[0],0]
        
        # if min_time_1.size > 1:
        #     min_time = min_time_1.iloc[0]
        #     start_time = min_time
        # else:
        #     start_time = min_time_1.to_string(index=False)
        
        start_time = str(min_time_1)
        
        #if type(start_time) == 'pandas.core.series.Series':
         #   start_time = str(min_time_1[0])
        
        # find time to turn device off
        tstamp = pd.Timestamp(start_time)
        hours_toadd = datetime.timedelta(hours = on_period*0.5)
        
        end_time_full = str(tstamp + hours_toadd)
        end_time = end_time_full[11:16]
        
        
        # Print time at which Z is minimum
        print('use op4')
        print('run 1')
        print('Start device at: ' + start_time)
        print('End device at: ' + end_time)
        
        # PRINT RESULTS TO CSV              
        csv_data = {'Device': [device_name],'Start Time': [start_time], 'End Time':[end_time]}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
        
    # Option 2    
    elif int(no_st_time_int[1:-1]) < int(deadline_int[1:-1]) < int(no_end_time_int[1:-1]):
        # use no_st_time_int as deadline
        
        if cut1_okay == False:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)

            sys.exit()
        
        # make first half of ci and pr        
        cut1_pr = pr_df_crop.iloc[:int(no_st_time_int[1:-1])-int(on_period),:]
        cut1_ci = ci_df_crop.iloc[:int(no_st_time_int[1:-1])-int(on_period),:]

        # make 2 new dataframes for comparison
        compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
        
        # apply optimisation 
        z1_1 = p*compare_df_1["Price"] # 1st column is price
        z2_1 = c*compare_df_1["CI"] # 2nd column is CI
        
        Z_1 = z1_1.add(z2_1)
        
        # Run optimisation for first split
        Z_data_1 = [compare_df_1["Time"], Z_1]
        Z_df_1 = pd.DataFrame(Z_data_1).transpose()
        
        a_1 = Z_df_1.index.values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_1) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:,0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        # find minimum time               
        y_min_1 = results_df_1[1].min()
        min_time_index_1 = results_df_1.index[results_df_1.iloc[:,1] == y_min_1]
        min_time_1 = results_df_1.iloc[min_time_index_1,0]
        
        if min_time_1.size > 1:
            min_time = min_time_1.iloc[0]
            start_time = min_time
        else:
            start_time = min_time_1.to_string(index=False)
    
        
        # find time to turn device off
        tstamp = pd.Timestamp(start_time)
        hours_toadd = datetime.timedelta(hours = on_period*0.5)
        
        end_time_full = str(tstamp + hours_toadd)
        end_time = end_time_full[11:16]
        
        
        # Print time at which Z is minimum
        print('use op4')
        print('run 2')
        print('Start device at: ' + start_time)
        print('End device at: ' + end_time)
        
        # PRINT RESULTS TO CSV              
        csv_data = {'Device': [device_name],'Start Time': [start_time], 'End Time':[end_time]}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
                                    
   
    # Option 3
    elif int(no_end_time_int[1:-1]) < int(deadline_int[1:-1]):
        # use deadline_int to cut off second half of 2nd cut
        
        # check whether 'cut1' is useable
        if cut1_okay == False:
            
            # only use second cut
            # make second half of ci and pr        
            cut2_pr = pr_df_crop.iloc[int(no_end_time_int[1:-1]):int(deadline_int[1:-1]),:]
            cut2_ci = ci_df_crop.iloc[int(no_end_time_int[1:-1]):int(deadline_int[1:-1]),:]

            # make new dataframe for comparison
            compare_df_2 = pd.DataFrame(zip(cut2_pr.iloc[:,0], cut2_pr.iloc[:,1], cut2_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
            
            # apply optimisation using only CUT 2
            z1_2 = p*compare_df_2["Price"] # 1st column is price
            z2_2 = c*compare_df_2["CI"] # 2nd column is CI
            
            Z_2 = z1_2.add(z2_2)
            
            # Run optimisation for second split
            Z_data_2 = [compare_df_2["Time"], Z_2]
            Z_df_2 = pd.DataFrame(Z_data_2).transpose()
            
            a_2 = Z_df_2.index[:-int(on_period)].values
            counter_2 = 0 # this is for the purposes of the for loop
            sum_results = []
            
            if len(a_2) == 0:
                print('No possible start time for given constraints')
                csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
                csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
                csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
                sys.exit()
            
            for each in a_2:
                # create list of values to sum
                position = a_2[counter_2]
                to_sum = Z_df_2.iloc[int(position):int(position+on_period), 1]
                #sum each possible scenario and save to results vector
                sum_results.append(sum(to_sum))
                counter_2 = counter_2 + 1
            
            results_2 = [Z_df_2.iloc[:-int(on_period),0].values, sum_results]
            results_df_2 = pd.DataFrame(results_2).transpose()
            
            # find minimum time               
            y_min_2 = results_df_2[1].min()
            min_time_index_2 = results_df_2.index[results_df_2.iloc[:,1] == y_min_2]
            min_time_2 = results_df_2.iloc[min_time_index_2,0]
            
            if min_time_2.size > 1:
                min_time = min_time_2.iloc[0]
                start_time_2 = min_time
            else:
                start_time_2 = min_time_2.to_string(index=False)
    
            
            # find time to turn device off
            tstamp = pd.Timestamp(start_time_2)
            hours_toadd = datetime.timedelta(hours = on_period*0.5)
            
            end_time_full = str(tstamp + hours_toadd)
            end_time = end_time_full[11:16]
            
            
            # Print time at which Z is minimum
            print('use op4')
            print('run 3 use cut 2')
            print('Start device at: ' + start_time_2)
            print('End device at: ' + end_time)
            
            # PRINT RESULTS TO CSV              
            csv_data = {'Device': [device_name],'Start Time': [start_time_2], 'End Time':[end_time]}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
           
        elif cut1_okay == True:
            # use cut1 and (cropped) cut2 as usual
        
            # make first half of ci and pr        
            cut1_pr = pr_df_crop.iloc[:int(no_st_time_int[1:-1])-int(on_period),:]
            cut1_ci = ci_df_crop.iloc[:int(no_st_time_int[1:-1])-int(on_period),:]
    
            # make second half of ci and pr
            cut2_pr = pr_df_crop.iloc[int(no_end_time_int[1:-1]):int(deadline_int[1:-1]),:]
            cut2_ci = ci_df_crop.iloc[int(no_end_time_int[1:-1]):int(deadline_int[1:-1]),:]
    
            # make 2 new dataframes for comparison
            compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
            compare_df_2 = pd.DataFrame(zip(cut2_pr.iloc[:,0], cut2_pr.iloc[:,1], cut2_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
            
            z1_1 = p*compare_df_1["Price"] # 1st column is price
            z2_1 = c*compare_df_1["CI"] # 2nd column is CI
            
            z1_2 = p*compare_df_2["Price"] # 1st column is price
            z2_2= c*compare_df_2["CI"] # 2nd column is CI
            
            Z_1 = z1_1.add(z2_1)
            Z_2 = z1_2.add(z2_2)
            
            # Run optimisation for first split
            Z_data_1 = [compare_df_1["Time"], Z_1]
            Z_df_1 = pd.DataFrame(Z_data_1).transpose()
            
            a_1 = Z_df_1.index.values
            counter_1 = 0 # this is for the purposes of the for loop
            sum_results = []
            
            if len(a_1) == 0:
                print('No possible start time for given constraints')
                csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
                csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
                csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
                sys.exit()
            
            for each in a_1:
                # create list of values to sum
                position = a_1[counter_1]
                to_sum = Z_df_1.iloc[int(position):int(position+on_period), 1]
                #sum each possible scenario and save to results vector
                sum_results.append(sum(to_sum))
                counter_1 = counter_1 + 1
            
            results_1 = [Z_df_1.iloc[:,0].values, sum_results]
            results_df_1 = pd.DataFrame(results_1).transpose()
            
            # Run optimisation for second split
            Z_data_2 = [compare_df_2["Time"], Z_2]
            Z_df_2 = pd.DataFrame(Z_data_2).transpose()
            
            a_2 = Z_df_2.index[:-int(on_period)].values
            counter_2 = 0 # this is for the purposes of the for loop
            sum_results = []
            
            if len(a_2) == 0:
                print('No possible start time for given constraints')
                csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
                csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
                csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
                sys.exit()
            
            for each in a_2:
                # create list of values to sum
                position = a_2[counter_2]
                to_sum = Z_df_2.iloc[int(position):int(position+on_period), 1]
                #sum each possible scenario and save to results vector
                sum_results.append(sum(to_sum))
                counter_2 = counter_2 + 1
            
            results_2 = [Z_df_2.iloc[:-int(on_period),0].values, sum_results]
            results_df_2 = pd.DataFrame(results_2).transpose()
            
            # find minimum time               
            y_min_1 = results_df_1[1].min()
            min_time_index_1 = results_df_1.index[results_df_1.iloc[:,1] == y_min_1]
            min_time_1 = results_df_1.iloc[min_time_index_1,0]
            
            if min_time_1.size > 1:
                min_time = min_time_1.iloc[0]
                start_time_1 = min_time
            else:
                start_time_1 = min_time_1.to_string(index=False)
    
            
            y_min_2 = results_df_2[1].min()
            min_time_index_2 = results_df_2.index[results_df_2.iloc[:,1] == y_min_2]
            min_time_2 = results_df_2.iloc[min_time_index_2,0]
            
            start_time_2 = min_time_2.to_string(index=False)
         
            # Get minimum out of 2 splits
            y_min = min(y_min_1, y_min_2)
            
            if y_min == y_min_1:
                start_time_1 = start_time_1
            elif y_min == y_min_2:
                start_time_1 = start_time_2
            
            # find time to turn device off
            tstamp = pd.Timestamp(start_time_1)
            hours_toadd = datetime.timedelta(hours = on_period*0.5)
            
            end_time_full = str(tstamp + hours_toadd)
            end_time = end_time_full[11:16]
            
            
            # Print time at which Z is minimum
            print('use op4')
            print('two split run')
            print('Start device at: ' + start_time_1)
            print('End device at: ' + end_time)
            
            # PRINT RESULTS TO CSV              
            csv_data = {'Device': [device_name],'Start Time': [start_time_1], 'End Time':[end_time]}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
           
    else:
        
        # apply optimisation 
        z1_1 = p*compare_df_1["Price"] # 1st column is price
        z2_1 = c*compare_df_1["CI"] # 2nd column is CI
        
        z1_2 = p*compare_df_2["Price"] # 1st column is price
        z2_2= c*compare_df_2["CI"] # 2nd column is CI
        
        Z_1 = z1_1.add(z2_1)
        Z_2 = z1_2.add(z2_2)
        
        # Run optimisation for first split
        Z_data_1 = [compare_df_1["Time"], Z_1]
        Z_df_1 = pd.DataFrame(Z_data_1).transpose()
        
        a_1 = Z_df_1.index[:-int(on_period)].values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_1) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:-int(on_period),0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        # Run optimisation for second split
        Z_data_2 = [compare_df_2["Time"], Z_2]
        Z_df_2 = pd.DataFrame(Z_data_2).transpose()
        
        a_2 = Z_df_2.index[:-int(on_period)].values
        counter_2 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        if len(a_2) == 0:
            print('No possible start time for given constraints')
            csv_data = {'Device': ['no start time'],'Start Time': [''], 'End Time':['']}
            csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
            csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
            sys.exit()
        
        for each in a_2:
            # create list of values to sum
            position = a_2[counter_2]
            to_sum = Z_df_2.iloc[int(position):int(position+on_period), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_2 = counter_2 + 1
        
        results_2 = [Z_df_2.iloc[:-int(on_period),0].values, sum_results]
        results_df_2 = pd.DataFrame(results_2).transpose()
        
        # find minimum time               
        y_min_1 = results_df_1[1].min()
        min_time_index_1 = results_df_1.index[results_df_1.iloc[:,1] == y_min_1]
        min_time_1 = results_df_1.iloc[min_time_index_1,0]
        
        if min_time_1.size > 1:
            min_time = min_time_1.iloc[0]
            start_time_1 = min_time
        else:
            start_time_1 = min_time_1.to_string(index=False)
    
        
        y_min_2 = results_df_2[1].min()
        min_time_index_2 = results_df_2.index[results_df_2.iloc[:,1] == y_min_2]
        min_time_2 = results_df_2.iloc[min_time_index_2,0]
        
        start_time_2 = min_time_2.to_string(index=False)
     
        # Get minimum out of 2 splits
        y_min = min(y_min_1, y_min_2)
        
        if y_min == y_min_1:
            start_time_1 = start_time_1
        elif y_min == y_min_2:
            start_time_1 = start_time_2
        
        # find time to turn device off
        tstamp = pd.Timestamp(start_time_1)
        hours_toadd = datetime.timedelta(hours = on_period*0.5)
        
        end_time_full = str(tstamp + hours_toadd)
        end_time = end_time_full[11:16]
        
        
        # Print time at which Z is minimum
        print('use op4')
        print('two split run')
        print('Start device at: ' + start_time_1)
        print('End device at: ' + end_time)
        
        # PRINT RESULTS TO CSV              
        csv_data = {'Device': [device_name],'Start Time': [start_time_1], 'End Time':[end_time]}
        csv_df = pd.DataFrame(csv_data, columns= ['Device', 'Start Time', 'End Time'])                          
        csv_df.to_csv (r'...\combined_output.csv', index = False, header=False)
       
#--------------------------------------------------------------------------------------------------------
# APPLY OPTIMISATIONS ACCORDING TO USER CONSTRAINTS
    
# Have 2 inputs, use_OFF and use_deadline 

if use_OFF == 'false' and use_deadline == 'false':
    use_op1()
        
elif use_OFF == 'true' and use_deadline == 'false':
    use_op2()

elif use_OFF == 'false' and use_deadline == 'true':
    use_op3()
    
elif use_OFF == 'true' and use_deadline == 'true':
    use_op4()

##############################################################################
# PLOTS for analysis
a = Z_df.index[:-int(on_period)].values
counter = 0
sum_results = []

for each in a:
    # create list of values to sum
    position = a[counter]
    to_sum = Z_df.iloc[int(position):int(position+on_period), 1]
    #sum each possible scenario and save to results vector
    sum_results.append(sum(to_sum))
    counter = counter + 1

results = [Z_df.iloc[:-int(on_period),0].values, sum_results]
results_df = pd.DataFrame(results).transpose()                                  

plot_results_df1 = results_df
plot_results_df1.iloc[:,1] = plot_results_df1.iloc[:,1]/abs(max(plot_results_df1.iloc[:,1]))


# PLOT Z_DF
plt.figure(1)
plt.plot(compare_df.iloc[:,0], compare_df.iloc[:,1], label='CI')
plt.plot(compare_df.iloc[:,0], compare_df.iloc[:,2], label= 'Price')
plt.plot(Z_df.iloc[:,0], Z_df.iloc[:,1], label='Z cost')
plt.plot(plot_results_df1.iloc[:,0], plot_results_df1.iloc[:,1], label='Integrated Cost', color='red')
#plt.plot(plot_results_df2.iloc[:,0], plot_results_df2.iloc[:,1], color='red')
plt.axvspan('00:00', '04:00', alpha=0.2, color='red', label='Device ON')
#plt.axvline('11:00', color='red', linestyle='--')
#plt.axvspan('13:00', '16:00', alpha=0.2, color='yellow', label='Must be OFF')
plt.xticks(Z_df.iloc[:,0][0::2], rotation=90)
plt.legend(loc='lower left')
plt.title('NO CONSTRAINTS, Price weight = 0.5, 16-03-2022')
plt.xlabel('Time, HH:MM')
plt.ylabel('Normalised cost')

plt.figure(4)
plt.plot(compare_df.iloc[:,0], compare_df.iloc[:,1], label='CI')
plt.plot(compare_df.iloc[:,0], compare_df.iloc[:,2], label= 'Price')
plt.plot(Z_df.iloc[:,0], Z_df.iloc[:,1], label='Z cost')
plt.plot(plot_results_df1.iloc[:,0], plot_results_df1.iloc[:,1], label='Integrated Cost', color='red')
#plt.plot(plot_results_df2.iloc[:,0], plot_results_df2.iloc[:,1], color='red')
#plt.axvspan('00:00', '04:00', alpha=0.2, color='red', label='Device ON')
#plt.axvline('11:00', color='red', linestyle='--')
#plt.axvspan('13:00', '16:00', alpha=0.2, color='yellow', label='Must be OFF')
plt.xticks(Z_df.iloc[:,0][0::2], rotation=90)
plt.legend(loc='lower left')
plt.title('Forecasts and Integrated Cost, Price weight = 0.5, 16-03-2022')
plt.xlabel('Time, HH:MM')
plt.ylabel('Normalised cost')

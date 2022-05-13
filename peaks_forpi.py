#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
import sys
import requests
import numpy as np

##########################################################################

# DEFINE FUNCTIONS TO GENERATE TEMPLATE CI AND PRICE FORECASTS

# generate a dataframe of times and CI values 
# parameters: length --> length of dataframes required, usually 48

def make_ci(length):
    # create list of times
    times = []
    for hour in range(24):
        for minute in range(0, 60, 30):
            times.append('{:02d}:{:02d}'.format(hour, minute))
            
    # Create list of optimisation variable
    import random
    randomlist1 = []
    for i in range(0,48):
        n = round(random.random(), 2)
        randomlist1.append(n)
    
    ci_data = {'Time':times[0:length], 'CI':randomlist1[0:length]}
    ci_df= pd.DataFrame(ci_data)
    
    return ci_df
        
def make_price(length):
    # create list of times
    times = []
    for hour in range(24):
        for minute in range(0, 60, 30):
            times.append('{:02d}:{:02d}'.format(hour, minute))
            
    # Create list of optimisation variable
    import random
    randomlist1 = []
    for i in range(0,48):
        n = round(random.random(), 2)
        randomlist1.append(n)
    
    price_data = {'Time':times[0:length], 'Price':randomlist1[0:length]}
    price_df= pd.DataFrame(price_data)
    
    return price_df        

#---------------------------------------------------------------------------
# DEFINE FUNCTIONS TO GET REAL CI DATA FROM API

def get_ci_data():
    
    #now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.datetime.now().strftime("%Y-%m-%d")

    co2_base_url = "https://api.carbonintensity.org.uk"
    #co2_url = (co2_base_url+"/intensity/"+now+"/fw24h")
    co2_url = (co2_base_url+"/intensity/" + now + "T05:00:00Z/fw24h")
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

    #now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    #now_fw24_str = now_fw24.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    now_str = now.strftime("%Y-%m-%d")
    now_fw24_str = now_fw24.strftime("%Y-%m-%d")

    url = ('https://api.octopus.energy/v1/products/AGILE-18-02-21/' + 
             'electricity-tariffs/E-1R-AGILE-18-02-21-C/standard-unit-rates/' + 
             '?period_from='+now_str+'T05:00:00Z&period_to='+now_fw24_str+'T05:00:00Z')
    
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

##########################################################################

# DEFINE FUNCTION TO GENERATE TEMPLATE DEVICE INFORMATION DATAFRAME        
        
# generate a dataframe of device information

def make_device_info():
    
    # Create fake data frames for dev
    column_names = ['device_name','power','on_period','use_OFF','s_off','e_off','use_deadline','deadline']
    
    # Here you can input the details you would like to include for devices
    # include data in the following formats
     
    #  | device_name | power   | on_period | use_OFF    | s_off | e_off | use_deadline | deadline |
    #  | string      | integer | integer   | true/false | HH:MM | HH:MM | true/false   | HH:MM    |
    
    column_names = ['device_name','power','on_period','use_OFF','s_off','e_off','use_deadline','deadline']
    d1_values = ['P2','2','2','true','15:00','17:00','true','10:00']
    d2_values = ['P3','7','6','true','16:00','19:00','true','22:00']
    d3_values = ['P4','3','12','true','11:00','12:00','true','17:30']

    d1_data_raw = [column_names, d1_values]
    d2_data_raw = [column_names, d2_values]
    d3_data_raw = [column_names, d3_values]

    d1_df_raw = pd.DataFrame(d1_data_raw)
    d2_df_raw = pd.DataFrame(d2_data_raw)
    d3_df_raw = pd.DataFrame(d3_data_raw)

    d1_df = d1_df_raw.transpose()
    d2_df = d2_df_raw.transpose()
    d3_df = d3_df_raw.transpose()

    device_data_all = [d1_df.iloc[:,0],d1_df.iloc[:,1], d2_df.iloc[:,1], d3_df.iloc[:,1]]
    device_info_df = pd.DataFrame(device_data_all).transpose()
    device_info_df.columns = [0,1,2,3]
        
    return device_info_df    


#---------------------------------------------------------------------------------------
# DEFINE FUNCTION TO GENERATE DEVICE INFORMATION DATAFRAME FROM NODE RED

# NEEDS WORK NOT READY YET

def get_device_info():
        
    device1_info = pd.read_csv(r'/home/pi/Documents/Working/device_1_info.csv', header=None)
    device1_df = pd.DataFrame(device1_info)
    start_OFF_dtime1 = str(pd.Timestamp(device1_df.iloc[4,1], unit='ms'))
    end_OFF_dtime1 = str(pd.Timestamp(device1_df.iloc[5,1], unit='ms'))
    device1_df.iloc[4,1] = start_OFF_dtime1[11:16]
    device1_df.iloc[5,1] = end_OFF_dtime1[11:16]
    
    deadline_raw1 = device1_df.iloc[7,1]
    deadline_long1 = str(pd.Timestamp(deadline_raw1, unit='ms'))
    device1_df.iloc[7,1] = deadline_long1[11:16]
    
    #-------------------------------------------------------------------------
    device2_info = pd.read_csv(r'/home/pi/Documents/Working/device_2_info.csv', header=None)
    device2_df = pd.DataFrame(device2_info)
    start_OFF_dtime2 = str(pd.Timestamp(device2_df.iloc[4,1], unit='ms'))
    end_OFF_dtime2 = str(pd.Timestamp(device2_df.iloc[5,1], unit='ms'))
    device2_df.iloc[4,1] = start_OFF_dtime2[11:16]
    device2_df.iloc[5,1] = end_OFF_dtime2[11:16]
    
    deadline_raw2 = device2_df.iloc[7,1]
    deadline_long2 = str(pd.Timestamp(deadline_raw2, unit='ms'))
    device2_df.iloc[7,1] = deadline_long2[11:16]
    
    #-------------------------------------------------------------------------
    device3_info = pd.read_csv(r'/home/pi/Documents/Working/device_3_info.csv', header=None)
    device3_df = pd.DataFrame(device3_info)
    start_OFF_dtime3 = str(pd.Timestamp(device3_df.iloc[4,1], unit='ms'))
    end_OFF_dtime3 = str(pd.Timestamp(device3_df.iloc[5,1], unit='ms'))
    device3_df.iloc[4,1] = start_OFF_dtime3[11:16]
    device3_df.iloc[5,1] = end_OFF_dtime3[11:16]
    
    deadline_raw3 = device3_df.iloc[7,1]
    deadline_long3 = str(pd.Timestamp(deadline_raw3, unit='ms'))
    device3_df.iloc[7,1] = deadline_long3[11:16]
    
    #-------------------------------------------------------------------------
    devices_df = device1_df
    devices_df['2'] = device2_df.iloc[:,1]
    devices_df['3'] = device3_df.iloc[:,1]
    
    return devices_df

###########################################################################
# DEFINE FUNCTION TO SET PRICE WEIGHTING

def make_pr_weight(weight):
    price_weight = weight
    return price_weight

#------------------------------------------------------------------------
# DEFINE FUNCTION TO GET PRICE WEIGHTING FROM NODE RED    

def get_pr_weight():
    price_weight = pd.read_csv(r'/home/pi/Documents/Working/price_weighting1.csv', header=None)
    price_weight = price_weight.iloc[0,0]
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

##############################################################################

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
    
################################################################################

# PREPARE DATA FOR OPTIMISATION

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


    
#########################################################################    

# DEFINE FUNCTIONS FOR OPTIMISATION

#------------------------------------------------------------------------
# OP1

def use_op1(device_no):
    
    a = Z_df.index[:-int(device_info_df.iloc[2,device_no])].values
    counter = 0
    sum_results = []

    for each in a:
        # create list of values to sum
        position = a[counter]
        to_sum = Z_df.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
        #sum each possible scenario and save to results vector
        sum_results.append(sum(to_sum))
        counter = counter + 1

    results = [Z_df.iloc[:-int(device_info_df.iloc[2, device_no]),0].values, sum_results]
    results_df = pd.DataFrame(results).transpose()                                  
    return results_df
   
#------------------------------------------------------------------------
# OP2

def use_op2(device_no):

    # get positional information: what index does no_device_start have in the 
    # normalised ci/pr lists?
    no_st_time_idx = ci_df_crop.index[ci_df_crop.iloc[:,0] == device_info_df.iloc[4,device_no]].tolist()
    no_end_time_idx = ci_df_crop.index[ci_df_crop.iloc[:,0] == device_info_df.iloc[5,device_no]].tolist()

    no_st_time_str = str(no_st_time_idx)
    no_end_time_str = str(no_end_time_idx)
    
    # make adjustment for if no end time in forecast list
    if no_st_time_str == '[]':
       no_st_time_str = str('[' + str(pr_df_crop.index[0]) + ']')
    if no_end_time_str == '[]':
        no_end_time_str = str('[' + str(pr_df_crop.index[-1]) + ']')

    # else split as usual and use 2 split run

    if int(no_st_time_str[1:-1]) < int(device_info_df.iloc[2,device_no]):
        
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
        
        a_2 = Z_df_2.index[:-int(device_info_df.iloc[2,device_no])].values
        counter_2 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        #if len(a_2) == 0:
            #no start time
        
        for each in a_2:
            # create list of values to sum
            position = a_2[counter_2]
            to_sum = Z_df_2.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_2 = counter_2 + 1
        
        results_2 = [Z_df_2.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
        results_df_2 = pd.DataFrame(results_2).transpose()
        return results_df_2
        
    elif int(pr_df_crop.index[-1]) - int(no_end_time_str[1:-1]) < int(device_info_df.iloc[2,device_no]):
        
        # not possible to make 2nd split. Only consider first split.
        # make cut at s_off - on_period
        
        # make first half of ci and pr
        cut1_pr = pr_df_crop.iloc[:int(no_st_time_str[1:-1])-int(device_info_df.iloc[2,device_no]),:]
        cut1_ci = ci_df_crop.iloc[:int(no_st_time_str[1:-1])-int(device_info_df.iloc[2,device_no]),:]
        
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
        
        #if len(a_1) == 0:
         #   NO POSSIBLE START TIMES
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:,0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        return results_df_1
        
    else:
        
        # split into two cuts and continue with optimisation
    
        # make first half of ci and pr
        cut1_pr = pr_df_crop.iloc[:int(no_st_time_str[1:-1])-int(device_info_df.iloc[2,device_no]),:]
        cut1_ci = ci_df_crop.iloc[:int(no_st_time_str[1:-1])-int(device_info_df.iloc[2,device_no]),:]
    
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
        
        a_1 = Z_df_1.index[:-int(device_info_df.iloc[2,device_no])].values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        #if len(a_1) == 0:
         #   NO POSSIBLE START TIMES
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        # Run optimisation for second split
        Z_data_2 = [compare_df_2["Time"], Z_2]
        Z_df_2 = pd.DataFrame(Z_data_2).transpose()
        
        a_2 = Z_df_2.index[:-int(device_info_df.iloc[2,device_no])].values
        counter_2 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        #if len(a_2) == 0:
         #   NO POSSIBLE START TIMES
        
        for each in a_2:
            # create list of values to sum
            position = a_2[counter_2]
            to_sum = Z_df_2.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_2 = counter_2 + 1
        
        results_2 = [Z_df_2.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
        results_df_2 = pd.DataFrame(results_2).transpose()
        
        results_df = results_df_1.append(results_df_2)
        
        return results_df

#------------------------------------------------------------------------
# OP3

def use_op3(device_no):
    
    # get positional information: what index does no_device_start have in the 
    # normalised ci/pr lists?
    deadline_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == device_info_df.iloc[7,device_no]].tolist()
    
    deadline_int = str(deadline_idx)
    
    #if deadline_idx == []:
     #   NO POSSIBLE START TIME
     
    #if int(deadline_int[1:-1])-int(device_info_df.iloc[2,device_no]) <= 0:
    #    NO POSSIBLE START TIME

    # make first half of ci and pr
    #cut1_pr = pr_df_crop.iloc[:int(deadline_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]
    #cut1_ci = ci_df_crop.iloc[:int(deadline_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]

    cut1_pr = pr_df_crop.iloc[:int(deadline_int[1:-1]),:]
    cut1_ci = ci_df_crop.iloc[:int(deadline_int[1:-1]),:]

    # make 2 new dataframes for comparison
    compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
    
    # check cut is long enough for on_period
    if len(cut1_pr) <= int(device_info_df.iloc[2,device_no]):
        print('No possible start time for given constraints')
        
    # apply optimisation 
    z1_1 = p*compare_df_1["Price"] # 1st column is price
    z2_1 = c*compare_df_1["CI"] # 2nd column is CI
    
    Z_1 = z1_1.add(z2_1)
    
    # Run optimisation for first split
    Z_data_1 = [compare_df_1["Time"], Z_1]
    Z_df_1 = pd.DataFrame(Z_data_1).transpose()
    
    a_1 = Z_df_1.index[:-int(device_info_df.iloc[2,device_no])].values
    counter_1 = 0 # this is for the purposes of the for loop
    sum_results = []
    
    #if len(a_1) == 0:
     #   NO POSSIBLE START TIMES
    
    for each in a_1:
        # create list of values to sum
        position = a_1[counter_1]
        to_sum = Z_df_1.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
        #sum each possible scenario and save to results vector
        sum_results.append(sum(to_sum))
        counter_1 = counter_1 + 1
    
    results_1 = [Z_df_1.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
    results_df_1 = pd.DataFrame(results_1).transpose()
    return results_df_1
     
#---------------------------------------------------------------------------------------------------
# OP4

def use_op4(device_no):
    
    # Get index positions of s_off, e_off and deadline
    deadline_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == device_info_df.iloc[7,device_no]].tolist()
    deadline_int = str(deadline_idx)
    
    no_st_time_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == device_info_df.iloc[4,device_no]].tolist()
    no_end_time_idx = pr_df_crop.index[pr_df_crop.iloc[:,0] == device_info_df.iloc[5,device_no]].tolist()

    no_st_time_int = str(no_st_time_idx)
    no_end_time_int = str(no_end_time_idx)
    
    # adjust no_st and no_end if necessary
    if no_st_time_int == '[]':
        no_st_time_int = str('[' + str(pr_df_crop.index[0]) + ']')
    
    if no_end_time_int == '[]':
        no_end_time_int = str('[' + str(pr_df_crop.index[-1]) + ']')
        
    # add tags to check whether cuts are long enough (note cut2 doesn't
    # require this as can project end time beyond forecast vector end)
    cut1_okay = True
    
    # is 'cut1' going to be long enough for device to run?
    if int(no_st_time_int[1:-1])-int(device_info_df.iloc[2,device_no]) <= 0:
        cut1_okay = False
    
    # Determine where the deadline sits relative to no_st and no_end
    # 1. deadline is before no_st
    # 2. deadline is between no_st and no_end
    # 3. deadline is after no_end
    
    # Option 1
    if int(deadline_int[1:-1]) <= int(no_st_time_int[1:-1]):
        # use deadline_int as deadline
        
        # check new dataframe is long enough for on_period
        if int(deadline_int[1:-1]) <= int(device_info_df.iloc[2,device_no]):
            print('No possible start time for given constraints')
            #csv_df.to_csv (r'/home/pi/Documents/Working/op_4_output.csv', index = False, header=False)
            sys.exit()
        
        # trim of ci and pr        
        cut1_pr = pr_df_crop.iloc[:int(deadline_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]
        cut1_ci = ci_df_crop.iloc[:int(deadline_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]

        # make 2 new dataframes for comparison
        compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
        
        # apply optimisation 
        z1_1 = p*compare_df_1["Price"] # 1st column is price
        z2_1 = c*compare_df_1["CI"] # 2nd column is CI
        
        Z_1 = z1_1.add(z2_1)
        
        # Run optimisation for first split
        Z_data_1 = [compare_df_1["Time"], Z_1]
        Z_df_1 = pd.DataFrame(Z_data_1).transpose()
        
        a_1 = Z_df_1.index[:-int(device_info_df.iloc[2,device_no])].values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        #if len(a_1) == 0:
         #   NO POSSIBLE START TIMES
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        return results_df_1
        
    # Option 2    
    elif int(no_st_time_int[1:-1]) < int(deadline_int[1:-1]) < int(no_end_time_int[1:-1]):
        # use no_st_time_int as deadline
        
        if cut1_okay == False:
            print('No possible start time for given constraints')
            #csv_df.to_csv (r'/home/pi/Documents/Working/op_4_output.csv', index = False, header=False)
            sys.exit()
        
        # make first half of ci and pr        
        cut1_pr = pr_df_crop.iloc[:int(no_st_time_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]
        cut1_ci = ci_df_crop.iloc[:int(no_st_time_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]

        # make new dataframe for comparison
        compare_df_1 = pd.DataFrame(zip(cut1_pr.iloc[:,0], cut1_pr.iloc[:,1], cut1_ci.iloc[:,1]), columns=["Time", "Price", "CI"])
        
        # apply optimisation 
        z1_1 = p*compare_df_1["Price"] # 1st column is price
        z2_1 = c*compare_df_1["CI"] # 2nd column is CI
        
        Z_1 = z1_1.add(z2_1)
        
        # Run optimisation for first split
        Z_data_1 = [compare_df_1["Time"], Z_1]
        Z_df_1 = pd.DataFrame(Z_data_1).transpose()
        
        a_1 = Z_df_1.index[:-int(device_info_df.iloc[2,device_no])].values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        #if len(a_1) == 0:
         #   NO POSSIBLE START TIMES
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        return results_df_1            
   
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
            
            a_2 = Z_df_2.index[:-int(device_info_df.iloc[2,device_no])].values
            counter_2 = 0 # this is for the purposes of the for loop
            sum_results = []
            
            #if len(a_2) == 0:
             #   NO POSSIBLE START TIMES
            
            for each in a_2:
                # create list of values to sum
                position = a_2[counter_2]
                to_sum = Z_df_2.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
                #sum each possible scenario and save to results vector
                sum_results.append(sum(to_sum))
                counter_2 = counter_2 + 1
            
            results_2 = [Z_df_2.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
            results_df_2 = pd.DataFrame(results_2).transpose()
            
            return results_df_2
            
        elif cut1_okay == True:
            # use cut1 and (cropped) cut2 as usual
        
            # make first half of ci and pr        
            cut1_pr = pr_df_crop.iloc[:int(no_st_time_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]
            cut1_ci = ci_df_crop.iloc[:int(no_st_time_int[1:-1])-int(device_info_df.iloc[2,device_no]),:]
    
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
            
            a_1 = Z_df_1.index[:-int(device_info_df.iloc[2,device_no])].values
            counter_1 = 0 # this is for the purposes of the for loop
            sum_results = []
            
            #if len(a_1) == 0:
             #   NO POSSIBLE START TIME
            
            for each in a_1:
                # create list of values to sum
                position = a_1[counter_1]
                to_sum = Z_df_1.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
                #sum each possible scenario and save to results vector
                sum_results.append(sum(to_sum))
                counter_1 = counter_1 + 1
            
            results_1 = [Z_df_1.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
            results_df_1 = pd.DataFrame(results_1).transpose()
            
            # Run optimisation for second split
            Z_data_2 = [compare_df_2["Time"], Z_2]
            Z_df_2 = pd.DataFrame(Z_data_2).transpose()
            
            a_2 = Z_df_2.index[:-int(device_info_df.iloc[2,device_no])].values
            counter_2 = 0 # this is for the purposes of the for loop
            sum_results = []
            
            #if len(a_2) == 0:
             #   NO POSSIBLE START TIMES
            
            for each in a_2:
                # create list of values to sum
                position = a_2[counter_2]
                to_sum = Z_df_2.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
                #sum each possible scenario and save to results vector
                sum_results.append(sum(to_sum))
                counter_2 = counter_2 + 1
            
            results_2 = [Z_df_2.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
            results_df_2 = pd.DataFrame(results_2).transpose()
            
            results_df = results_df_1.append(results_df_2)
            return results_df
            
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
        
        a_1 = Z_df_1.index[:-int(device_info_df.iloc[2,device_no])].values
        counter_1 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        #if len(a_1) == 0:
         #   NO POSSIBLE START TIMES
        
        for each in a_1:
            # create list of values to sum
            position = a_1[counter_1]
            to_sum = Z_df_1.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_1 = counter_1 + 1
        
        results_1 = [Z_df_1.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
        results_df_1 = pd.DataFrame(results_1).transpose()
        
        # Run optimisation for second split
        Z_data_2 = [compare_df_2["Time"], Z_2]
        Z_df_2 = pd.DataFrame(Z_data_2).transpose()
        
        a_2 = Z_df_2.index[:-int(device_info_df.iloc[2,device_no])].values
        counter_2 = 0 # this is for the purposes of the for loop
        sum_results = []
        
        #if len(a_2) == 0:
         #   NO POSSIBLE START TIMES
        
        for each in a_2:
            # create list of values to sum
            position = a_2[counter_2]
            to_sum = Z_df_2.iloc[int(position):int(position)+int(device_info_df.iloc[2,device_no]), 1]
            #sum each possible scenario and save to results vector
            sum_results.append(sum(to_sum))
            counter_2 = counter_2 + 1
        
        results_2 = [Z_df_2.iloc[:-int(device_info_df.iloc[2,device_no]),0].values, sum_results]
        results_df_2 = pd.DataFrame(results_2).transpose()
        
        results_df = results_df_1.append(results_df_2)
        return results_df

########################################################################################################

# GET COST VARIABLE FOR EACH DEVICE USING APPROPRIATE OPTIMISATION 

if device_info_df.iloc[3,1] == 'false' and device_info_df.iloc[6,1] == 'false':
    Z1_df = use_op1(1)            
elif device_info_df.iloc[3,1] == 'true' and device_info_df.iloc[6,1] == 'false':
    Z1_df = use_op2(1)
elif device_info_df.iloc[3,1] == 'false' and device_info_df.iloc[6,1] == 'true':
    Z1_df = use_op3(1)        
elif device_info_df.iloc[3,1] == 'true' and device_info_df.iloc[6,1] == 'true':
    Z1_df = use_op4(1)  
    
if device_info_df.iloc[3,2] == 'false' and device_info_df.iloc[6,2] == 'false':
    Z2_df = use_op1(2)            
elif device_info_df.iloc[3,2] == 'true' and device_info_df.iloc[6,2] == 'false':
    Z2_df = use_op2(2)
elif device_info_df.iloc[3,2] == 'false' and device_info_df.iloc[6,2] == 'true':
    Z2_df = use_op3(2)        
elif device_info_df.iloc[3,2] == 'true' and device_info_df.iloc[6,2] == 'true':
    Z2_df = use_op4(2)  

if device_info_df.iloc[3,3] == 'false' and device_info_df.iloc[6,3] == 'false':
    Z3_df = use_op1(3)            
elif device_info_df.iloc[3,3] == 'true' and device_info_df.iloc[6,3] == 'false':
    Z3_df = use_op2(3)
elif device_info_df.iloc[3,3] == 'false' and device_info_df.iloc[6,3] == 'true':
    Z3_df = use_op3(3)        
elif device_info_df.iloc[3,3] == 'true' and device_info_df.iloc[6,3] == 'true':
    Z3_df = use_op4(3)      


########################################################################################################

# APPLY PEAK AVOIDANCE

# Set a max limit of power
power_limit = 4

################################################################################
# COMBINE COSTS FOR EACH DEVICE
# make new dataframe with following format:
# index | Time (from Z_df) | Device1 costs | Device2 costs | Device3 costs

combo_df = pd.DataFrame(Z_df.iloc[:,0])
device_nos = list(range(1, len(device_info_df.columns)))

# get Z_df index of times that are in Z1_df too
Z1_times = []
for each in Z1_df.iloc[:,0]:
    if each in combo_df.iloc[:,0].values:
        Z1_times.append(each)
combo_df['1'] = [float(power_limit + 1)]*len(combo_df)
for each in list(range(len(Z1_times))):
    time_idx = combo_df[combo_df.iloc[:,0] == Z1_times[each]].index.tolist()
    time_idx = time_idx[0]
    combo_df['1'][time_idx] = float(Z1_df.iloc[each,1])

Z2_times = []
for each in Z2_df.iloc[:,0]:
    if each in combo_df.iloc[:,0].values:
        Z2_times.append(each)
combo_df['2'] = [float(power_limit + 1)]*len(combo_df)
for each in list(range(len(Z2_times))):
    time_idx = combo_df[combo_df.iloc[:,0] == Z2_times[each]].index.tolist()
    time_idx = time_idx[0]
    combo_df['2'][time_idx] = float(Z2_df.iloc[each,1])
    
Z3_times = []
for each in Z3_df.iloc[:,0]:
    if each in combo_df.iloc[:,0].values:
        Z3_times.append(each)
combo_df['3'] = [float(power_limit + 1)]*len(combo_df)
for each in list(range(len(Z3_times))):
    time_idx = combo_df[combo_df.iloc[:,0] == Z3_times[each]].index.tolist()
    time_idx = time_idx[0]
    combo_df['3'][time_idx] = float(Z3_df.iloc[each,1])    



#####################################################################################

# PLACEMENT OF COST BLOCKS
# use For loop to iterate through this list of devices in order, placing
# each at best permissible space

# 1. find best place for largest cost to go. 
# 2. find best place for 2nd largest cost to go
# 3. find best place for 3rd largest cost to go
# 4. find best place for nth largest cost to go

# for each device, ordered by size of Z_min cost

Z1_df_ordered = Z1_df.sort_values(1)
Z2_df_ordered = Z2_df.sort_values(1)
Z3_df_ordered = Z3_df.sort_values(1)


def place_device(device_no, power_list):
    
    # make nZ_df list
    nZ_df = combo_df.iloc[:, [0, device_no]]
    # order nZ_df list
    nZ_df_ordered = nZ_df.sort_values(str(device_no))
    
    # truncate to last time that device could start
    nZ_df_ordered = nZ_df_ordered[nZ_df_ordered.index <= int(len(combo_df)) - int(device_info_df.iloc[2,device_no])]
    
    # add a new index to the df
    new_index = list(range(len(nZ_df_ordered)))
    
    nZ_df_ordered['new index'] = new_index

    pwr_list_toadd = [int(device_info_df.iloc[1,device_no])]*int(device_info_df.iloc[2,device_no])
    
    counter = 0
    
    while counter <= len(nZ_df_ordered):
        # for each* in Z_df_ordered list
        device_on_period = int(device_info_df.iloc[2, device_no])
        
        index_start_on = nZ_df_ordered.index[counter]
        
        # add the power vector for device to the master power vector
        power_local = np.array(power_list)
        #power_local[index_start_on:index_end_on+1] = power_local[index_start_on:index_end_on+1] + np.array(pwr_list_toadd)
        power_local[index_start_on:index_start_on+device_on_period] = power_local[index_start_on:index_start_on+device_on_period] + np.array(pwr_list_toadd)
        power_local = list(power_local)
        
        # check that all values in power vector are below power limit
        if counter >= int(len(combo_df))-int(device_info_df.iloc[2,device_no]):
            return([0]*48, 'no start time')
            break
        elif all(i <= power_limit for i in power_local) == True:
            return power_local, nZ_df_ordered.iloc[counter,0]
            break
        elif all(i <= power_limit for i in power_local) == False:
            # return power list to how it was 
            power_local = power_list
            counter = counter + 1
            continue 
        
# now place each device sequentially
power_blank = [0]*len(combo_df)
power_master = power_blank

# want to find out which devices have largest => smallest Z costs
x = [int(device_info_df.iloc[1,1])*int(device_info_df.iloc[2,1]),int(device_info_df.iloc[1,2])*int(device_info_df.iloc[2,2]),int(device_info_df.iloc[1,3])*int(device_info_df.iloc[2,3])]
y = pd.DataFrame(x)

device_nos = list(range(1, len(device_info_df.columns)))
values = [y.iloc[0,0], y.iloc[1,0], y.iloc[2,0]]
y['Value'] = values
y['Device_no'] = device_nos

y_sorted = y.sort_values('Value', ascending=False)

device_ordered_costs = list(y_sorted.iloc[:,2].values)
# this is simply a list of device numbers, ordered by their largest Z cost

# make output dataframe of device number and start/end times
start_times = []

for each in device_ordered_costs:
    
    if place_device(each, power_master)[1] == 'no time to start device':
        print('cannot place device no' + str(each))
        start_times.append('no start time')
    
    [power_master, start_time] = place_device(each, power_master) 

    start_times.append(start_time)    

    # make power limit vector to plot
    pwr_limit = [power_limit]*len(combo_df)
    
    plt.figure(each)
    plt.plot(pwr_limit)
    plt.plot(combo_df.iloc[:,0],power_master)
    plt.xlabel('1/2 hr period')
    plt.ylabel('Power W')
    plt.xticks(rotation=90)
    plt.xticks(np.arange(0, len(combo_df.iloc[:,0])+1, 4))
    plt.title('Power Forecast')
   
    if all(i <= power_limit for i in list(power_master)) == True:
        continue
    elif all(i <= power_limit for i in list(power_master)) == False:
        print('cannot place this device, no.' + str(each))
        print('end op.')
        break
    
output_data = zip(device_ordered_costs, start_times)
output_df = pd.DataFrame(output_data)


# want output csvs... one for each device... start and end time
# get end times from combo_df and jump forward on_period no of intervals

end_times = []
for each in list(range(len(output_df))):
    device_period = int(device_info_df.iloc[2, each+1])
    if output_df.iloc[each, 1] == 'no start time':
        end_times.append('no end time')
    else:
        tstamp = pd.Timestamp(output_df.iloc[each, 1])
        hours_toadd = datetime.timedelta(hours = device_period*0.5)
        end_time_full = str(tstamp + hours_toadd)
        end_time = end_time_full[11:16]
        end_times.append(end_time)

output_df['end times'] = end_times
csv_df = output_df
csv_df.to_csv (r'/home/pi/Documents/Codes/Peak Avoidance/pk_output_times.csv', index = False, header=False)

print(output_df)   
   






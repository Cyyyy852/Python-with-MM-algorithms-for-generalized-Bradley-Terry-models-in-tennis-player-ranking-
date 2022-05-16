# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:25:33 2021

@author: s1155125208
"""

import pandas as pd
import numpy as np
import copy
import math

    

#ATP = pd.read_csv(r'C:\Users\s1155125208\Desktop\combined data.csv')  # choose suitable directory
ATP = pd.read_csv(r'C:\Users\Lai Tsz Chun\OneDrive\Documents\combined data.csv')  # choose suitable directory
estimation_all = []  #save all estimates in each timeframe
Players_li = []
rank_pred_dic = []
estimation_all_2 = []  #save all estimates in each timeframe
rank_pred_dic_2 = []
estimation_all_3 = []  #save all estimates in each timeframe
rank_pred_dic_3 = []

nam_wl = ['W1','L1','W2','L2','W3','L3','W4','L4','W5','L5']

data_entire = ATP[ATP['Comment'] == 'Completed']
data_entire = data_entire.fillna(0)
for i in range(2000,2018):
    data = []
    for j in range(5):
        data_temp = data_entire[data_entire['Year'] == i+j]
        data.append(data_temp)
    data = pd.concat(data)
    
    winner = np.unique(data['Winner'])
    loser =  np.unique(data['Loser'])
   
    
    Players = np.unique(np.intersect1d(winner,loser))
    num_player = len(Players)
    num_player_2 = len(Players)
    num_player_3 = len(Players)
    dic = dict(zip(Players, range(num_player)))
    Players_li.append(Players)
    
    
    win_graph = np.zeros((num_player,num_player))
    win_graph_2 = np.zeros((num_player_2,num_player_2))
    win_graph_3 = np.zeros((num_player_3,num_player_3))
    
    
    for k in range(len(data)):
        current_data = data.iloc[k]
        current_winner, current_loser = current_data['Winner'], current_data['Loser']
        for i in nam_wl:
            if current_data[i] == ' ':
                current_data[i] = 0
        if (current_winner in Players) and (current_loser in Players):
            current_win_index, current_lose_index = dic[current_winner], dic[current_loser]
            win_graph[current_win_index, current_lose_index] += 1
            win_graph_2[current_win_index, current_lose_index] += current_data['Wsets']
            win_graph_2[current_lose_index, current_win_index] += current_data['Lsets']
            win_graph_3[current_win_index, current_lose_index] += (float(current_data['W1'])+float(current_data['W2'])+float(current_data['W3'])+float(current_data['W4'])+float(current_data['W5']))
            win_graph_3[current_lose_index, current_win_index] += (float(current_data['L1'])+float(current_data['L2'])+float(current_data['L3'])+float(current_data['L4'])+float(current_data['L5']))
            

# compute MLE: delete some data
    for _ in range(1000):
        win = np.sum(win_graph,axis=1)
        lose = np.sum(win_graph, axis = 0)
        outlier = np.union1d(np.where(win==0)[0], np.where(lose==0)[0])    #union the 0 wins or 0 loss in a tuple
        index = np.setdiff1d(np.array((range(num_player))),outlier)       #Return the unique values in ar1 that are not in ar2
        win_graph = (win_graph[:,index])[index,:]
        num_player = len(index)
        if outlier.size == 0:
            break
        
    for _ in range(1000):
        win = np.sum(win_graph_2,axis=1)
        lose = np.sum(win_graph_2, axis = 0)
        outlier = np.union1d(np.where(win==0)[0], np.where(lose==0)[0])    #union the 0 wins or 0 loss in a tuple
        index = np.setdiff1d(np.array((range(num_player_2))),outlier)       #Return the unique values in ar1 that are not in ar2
        win_graph_2 = (win_graph_2[:,index])[index,:]
        num_player_2 = len(index)
        if outlier.size == 0:
            break
    
    for _ in range(1000):
        win = np.sum(win_graph_3,axis=1)
        lose = np.sum(win_graph_3, axis = 0)
        outlier = np.union1d(np.where(win==0)[0], np.where(lose==0)[0])    #union the 0 wins or 0 loss in a tuple
        index = np.setdiff1d(np.array((range(num_player_3))),outlier)       #Return the unique values in ar1 that are not in ar2
        win_graph_3 = (win_graph_3[:,index])[index,:]
        num_player_3 = len(index)
        if outlier.size == 0:
            break
        
        
    initial = np.ones(num_player)
    iteration = 1000
    win = np.sum(win_graph,axis=1)
    comparison_graph = np.transpose(win_graph) + win_graph
    
    initial_2 = np.ones(num_player_2)
    win_2 = np.sum(win_graph_2,axis=1)
    comparison_graph_2 = np.transpose(win_graph_2) + win_graph_2
    
    initial_3 = np.ones(num_player_3)
    win_3 = np.sum(win_graph_3,axis=1)
    comparison_graph_3 = np.transpose(win_graph_3) + win_graph_3

    for itr in range(iteration):
        last = initial.copy()
        current_probability = np.zeros((num_player,num_player))
        
        for m in range(num_player):
            current_probability[m] = initial[m] + initial  # Some correction
            
        g_matrix = comparison_graph/(current_probability)   # Some correction
        g_matrix_sum = np.sum(g_matrix,axis=1)        
        initial = (win)/g_matrix_sum
        initial = initial/initial[0]        # choose one as the benchmark

        if np.max(np.abs(last - initial)) < 1e-5:  # Converage criterion
            print('Pairwise Converge')
            estimation = initial
            print(estimation)
            break
        
    for itr in range(iteration):
        last = initial_2.copy()
        current_probability = np.zeros((num_player_2,num_player_2))
        
        for m in range(num_player_2):
            current_probability[m] = initial_2[m] + initial_2  # Some correction
            
        g_matrix = comparison_graph_2/(current_probability)   # Some correction
        g_matrix_sum = np.sum(g_matrix,axis=1)        
        initial_2 = (win_2)/g_matrix_sum
        initial_2 = initial_2/initial_2[0]        # choose one as the benchmark

        if np.max(np.abs(last - initial_2)) < 1e-5:  # Converage criterion
            print('Pairwise Converge')
            estimation_2 = initial_2
            print(estimation_2)
            break
    
    for itr in range(iteration):
        last = initial_3.copy()
        current_probability = np.zeros((num_player_3,num_player_3))
        
        for m in range(num_player_3):
            current_probability[m] = initial_3[m] + initial_3  # Some correction
            
        g_matrix = comparison_graph_3/(current_probability)   # Some correction
        g_matrix_sum = np.sum(g_matrix,axis=1)        
        initial_3 = (win_3)/g_matrix_sum
        initial_3 = initial_3/initial_3[0]        # choose one as the benchmark

        if np.max(np.abs(last - initial_3)) < 1e-5:  # Converage criterion
            print('Pairwise Converge')
            estimation_3 = initial_3
            print(estimation_3)
            break
        
    estimation_all.append(estimation)
    dic_pred = dict(zip(Players, estimation))
    rank_pred_dic.append(dic_pred)  #Store each timeframe player's gamma
    
    estimation_all_2.append(estimation_2)
    dic_pred_2 = dict(zip(Players, estimation_2))
    rank_pred_dic_2.append(dic_pred_2)  #Store each timeframe player's gamma
    
    estimation_all_3.append(estimation_3)
    dic_pred_3 = dict(zip(Players, estimation_3))
    rank_pred_dic_3.append(dic_pred_3)  #Store each timeframe player's gamma
        
#########################################################################################################################
### Running Windows

neg_log = []
neg_log_2 = []
neg_log_3 = []
actwin = []
for i in range(17):
    actual_data = data_entire[data_entire['Year'] == 2005 + i]
    winner = np.unique(actual_data['Winner'])
    loser =  np.unique(actual_data['Loser'])
    Players = np.unique(np.intersect1d(winner,loser))
    
    temp_players = rank_pred_dic[i].copy()
    temp_players_2 = rank_pred_dic_2[i].copy()
    temp_players_3 = rank_pred_dic_3[i].copy()
    
    for eachplayer in rank_pred_dic[i]:
        if eachplayer not in Players:
            del temp_players[eachplayer]
    
    for eachplayer in rank_pred_dic_2[i]:
        if (eachplayer not in rank_pred_dic[i]) or (eachplayer not in Players):
            del temp_players_2[eachplayer]
            
    for eachplayer in rank_pred_dic_3[i]:
        if (eachplayer not in Players) or (eachplayer not in rank_pred_dic[i]):
            del temp_players_3[eachplayer]
        
            
    actnum = len(temp_players)
    actnum_2 = len(temp_players_2)
    actnum_3 = len(temp_players_3)
    
    actwin_graph = np.zeros((actnum,actnum))
    dic = dict(zip(list(temp_players.keys()), range(actnum)))
    
    actwin_graph_2 = np.zeros((actnum_2,actnum_2))
    dic_2 = dict(zip(list(temp_players_2.keys()), range(actnum_2)))
    
    actwin_graph_3 = np.zeros((actnum_3,actnum_3))
    dic_3 = dict(zip(list(temp_players_3.keys()), range(actnum_3)))
    
    
    for m in range(len(actual_data)):
        current_actdata = actual_data.iloc[m]
        current_actwinner, current_actloser = current_actdata['Winner'], current_actdata['Loser']
        
        for i in nam_wl:
            if current_actdata[i] == ' ':
                current_actdata[i] = 0
                
        if (current_actwinner in list(temp_players.keys())) and (current_actloser in list(temp_players.keys())):
            current_actwin_index, current_actlose_index = dic[current_actwinner], dic[current_actloser]
            actwin_graph[current_actwin_index, current_actlose_index] += 1
            
        if (current_actwinner in list(temp_players_2.keys())) and (current_actloser in list(temp_players_2.keys())):
            current_actwin_index, current_actlose_index = dic_2[current_actwinner], dic_2[current_actloser]
            actwin_graph_2[current_actwin_index, current_actlose_index] += current_actdata['Wsets']
            actwin_graph_2[current_actlose_index, current_actwin_index] += current_actdata['Lsets']
        
        if (current_actwinner in list(temp_players_3.keys())) and (current_actloser in list(temp_players_3.keys())):
            current_actwin_index, current_actlose_index = dic_3[current_actwinner], dic_3[current_actloser]
            actwin_graph_3[current_actwin_index, current_actlose_index] += (float(current_actdata['W1'])+float(current_actdata['W2'])+float(current_actdata['W3'])+float(current_actdata['W4'])+float(current_actdata['W5']))
            actwin_graph_3[current_actlose_index, current_actwin_index] += (float(current_actdata['L1'])+float(current_actdata['L2'])+float(current_actdata['L3'])+float(current_actdata['L4'])+float(current_actdata['L5']))
            
    actwin.append(actwin_graph)
            
    neg_loglikelihood_all = 0
    for eachi in range(actnum):
        for eachj in range(actnum):
            neg_loglikelihood_temp = actwin_graph[eachi, eachj]*math.log(list(temp_players.values())[eachi]/(list(temp_players.values())[eachi] + list(temp_players.values())[eachj]))
            neg_loglikelihood_all += neg_loglikelihood_temp
            
    neg_loglikelihood_all_2 = 0
    for eachi in range(actnum_2):
        for eachj in range(actnum_2):
            p = list(temp_players_2.values())[eachi]/((list(temp_players_2.values())[eachi]) + list(temp_players_2.values())[eachj])
            neg_loglikelihood_temp = actwin_graph[eachi, eachj]*math.log(4*p**2 - 3*p**3)
            neg_loglikelihood_all_2 += neg_loglikelihood_temp
    
    neg_loglikelihood_all_3 = 0
    for eachi in range(actnum_3):
        for eachj in range(actnum_3):
            p = list(temp_players_3.values())[eachi]/((list(temp_players_3.values())[eachi]) + list(temp_players_3.values())[eachj])
            p_set = 1716*(p-1)**6*p**7+(210*p**4-924*p**3+1540*p**2-1154*p+329)*p**6-(p-1)**6-792*(p-1)**5
            neg_loglikelihood_temp = actwin_graph[eachi,eachj]*math.log(4*p**2 - 3*p**3)
            neg_loglikelihood_all_3 += neg_loglikelihood_temp
            
    neg_log_2.append(-neg_loglikelihood_all_2)  #negative log likelihood of each running windows 
    
    neg_log_3.append(-neg_loglikelihood_all_3)  #negative log likelihood of each running windows 
    
    neg_log.append(-neg_loglikelihood_all)  #negative log likelihood of each running windows 
           
            
### implied probability

bet_neg_log = []
for i in range(17):
    bet_data = data_entire[data_entire['Year'] == 2005+i]
    bet365_winner = bet_data['B365W']
    bet365_loser = bet_data['B365L']
    implied_win365 = (1/bet365_winner)
    implied_lose365 = (1/bet365_loser)
    norm_win365 = list(implied_win365/ (implied_win365 + implied_lose365))
    norm_lose365 = list(implied_lose365/ (implied_win365 + implied_lose365))
    bet_neg_log_all = 0
    
    for j in range(len(norm_win365)):
        if (math.isnan(norm_win365[j])) or (math.isnan(norm_lose365[j])):
            bet_neg_log_temp = 0
        else:
            bet_neg_log_temp = math.log(norm_win365[j])
        bet_neg_log_all += bet_neg_log_temp
    
    bet_neg_log.append(-bet_neg_log_all)
           

#########################
#########################
#Result
    
print(neg_log)  #BT model
print(neg_log_2)  # BT model using winning sets
print(neg_log_3)  # BT model using winning games
print(bet_neg_log)  #bet negative likelihood

y = range(2005,2022)
d = {'year':y, 'BT':neg_log, 'BT_sets':neg_log_2, 'BT_games':neg_log_3,'bet':bet_neg_log}
d = pd.DataFrame(d)
print(d)


#########################################################################################################################
### Difference of the estimation of different surfaces

surface_est = []
surface_dic = []
surface = ['Hard', 'Grass', 'Clay', 'Carpet']
for eachsurface in surface:
    data = data_entire[data_entire['Surface'] == eachsurface]
    
    winner = np.unique(data['Winner'])
    loser =  np.unique(data['Loser'])
   
    Players = np.unique(np.intersect1d(winner,loser))
    num_player = len(Players)
    dic = dict(zip(Players, range(num_player)))
    
    win_graph = np.zeros((num_player,num_player))
    
    for k in range(len(data)):
        current_data = data.iloc[k]
        current_winner, current_loser = current_data['Winner'], current_data['Loser']
        if (current_winner in Players) and (current_loser in Players):
            current_win_index, current_lose_index = dic[current_winner], dic[current_loser]
            win_graph[current_win_index, current_lose_index] += 1
    
    # compute MLE: delete some data
    for _ in range(1000):
        win = np.sum(win_graph,axis=1)
        lose = np.sum(win_graph, axis = 0)
        outlier = np.union1d(np.where(win==0)[0], np.where(lose==0)[0])    #union the 0 wins or 0 loss in a tuple
        index = np.setdiff1d(np.array((range(num_player))),outlier)       #Return the unique values in ar1 that are not in ar2
        win_graph = (win_graph[:,index])[index,:]
        num_player = len(index)
        if outlier.size == 0:
            break
        
        
    initial = np.ones(num_player)
    iteration = 1000
    win = np.sum(win_graph,axis=1)
    comparison_graph = np.transpose(win_graph) + win_graph

    for itr in range(iteration):
        last = initial.copy()
        current_probability = np.zeros((num_player,num_player))
        for m in range(num_player):
            current_probability[m] = initial[m] + initial  # Some correction
        
        g_matrix = comparison_graph/(current_probability)   # Some correction
        g_matrix_sum = np.sum(g_matrix,axis=1)
        
        initial = (win)/g_matrix_sum
        initial = initial/initial[0]        # choose one as the benchmark
        if np.max(np.abs(last - initial)) < 1e-5:  # Converage criterion
            print('Pairwise Converge')
            estimation = initial
            print(estimation)
            break
        
    surface_est.append(estimation)
    temp_dic = dict(zip(Players, estimation))
    surface_dic.append(temp_dic)

surface_key = []

for each in surface_dic:
    surface_key.extend(list(each.keys()))
    
surface_key = np.unique(surface_key)

temp_arr = np.zeros((len(surface_key),len(surface)))

for i in range(len(surface_key)):
    temp = []
    key = surface_key[i]
    for eachsurface in surface_dic:
        temp.append(eachsurface[key]) if key in eachsurface else temp.append(0)  #assign 0 if the player haven't played on that surface
    temp_arr[i] = temp

df_surface = pd.DataFrame(temp_arr, columns = surface )
df_surface['Player'] = surface_key    
df_surface.to_csv('Player_surface.csv', encoding='utf-8', index=False)

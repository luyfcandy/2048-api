# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 23:30:13 2018

@author: luyfc
"""
# python onlinetrain2_2.py

import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,TensorDataset
#from model import SimpleNet
from modelv3_0 import SimpleNet3
from game2048.game import Game
from game2048.displays import Display
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent,DIYAgent,DIY2Agent,DIY21Agent
from game2048.expectimax import board_to_move
from tqdm import tqdm

#torch.set_default_tensor_type('torch.DoubleTensor')
Batch_size=2000

OUT_SHAPE=(4,4)
CAND=12
map_table={2**i: i for i in range(1,CAND)}
map_table[0]=0
   
def grid_ohe(lst):
    arr=lst.reshape(4,4)
    ret=np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,map_table[arr[r,c]]]=1
    ret=np.swapaxes(ret,0,2)
    ret=np.swapaxes(ret,1,2)
    return ret

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score
rate_history=[5]
nicenet=[0]

train_num=20000000
position=0
capacity=720000

for i in range(67,train_num):
    print('loading data...')
    print(i)
    if(i==67):
        alldata=np.loadtxt('b128-'+str(i)+'.csv',usecols=(0,1,2,3,4,5,6,7,8,9,10,
                                           11,12,13,14,15,16),delimiter=",")
        np.random.shuffle(alldata) 
        datasets1=alldata[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
        datasets=alldata[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
        labels=alldata[:,16]
        labels=labels.astype(int)

        if(len(datasets)>capacity):
            datasets1=datasets1[0:capacity]
            datasets=datasets[0:capacity]
    
        if(len(labels)>capacity):
            labels=labels[0:capacity]
        position=len(labels)
        test_datasets=np.loadtxt('b128-test.csv',usecols=(0,1,2,3,4,5,6,7,8,9,10,
                                           11,12,13,14,15),delimiter=",")
        test_labels=np.loadtxt('b128-test.csv',usecols=16,delimiter=",")
        test_labels=test_labels.astype(int)
        test_onehot=np.empty(shape=[len(test_labels),12,4,4])
		
        data_onehot=np.empty(shape=[len(labels),12,4,4])
        pros=tqdm(datasets)
        iii=0
        iiii=0
        for (idxt,item) in enumerate(pros):
            tmp=grid_ohe(item)
            data_onehot[iii]=tmp
            iii=iii+1
            #data_onehot=np.append(data_onehot,[tmp],axis=0)
        for itemt in test_datasets:
            tmpt=grid_ohe(itemt)
            test_onehot[iiii]=tmpt
            iiii=iiii+1
            #test_onehot=np.append(test_onehot,[tmpt],axis=0)
    if(i>67):
        #tmpdata=np.loadtxt('b2048-'+str(i)+'.csv',usecols=(0,1,2,3,4,5,6,7,8,9,10,
                                           #11,12,13,14,15,16),delimiter=",")
        tmpdata=saveboard
        for itemtmp in tmpdata:
            #tmp=grid_ohe(itemtmp)
            position=position % capacity
            datasets1[position]=itemtmp
            position=position+1
            #data_onehot[position]=tmp
        tmpdatasets=datasets1[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
        
        tmplabels=datasets1[:,16]
        tmplabels=tmplabels.astype(int)
        labels=tmplabels
        
        if(i%8==0):
            np.savetxt('newboard_'+str(i)+'.csv',datasets, delimiter=',')
		
        data_onehot=np.empty(shape=[len(tmplabels),12,4,4])
        pros=tqdm(tmpdatasets)
        i6=0
        
        for (idxt,item) in enumerate(pros):
            tmp=grid_ohe(item)
            data_onehot[i6]=tmp
            i6=i6+1
            #data_onehot=np.append(data_onehot,[tmp],axis=0)
        

    print('loading..'+str(len(data_onehot)))
    
    
    train_data=torch.from_numpy(data_onehot) 
    train_label=torch.from_numpy(labels)
    test_data=torch.from_numpy(test_onehot) 
    test_label=torch.from_numpy(test_labels)
    print('loading...'+str(position))
    
    deal_train_dataset=TensorDataset(train_data,train_label)
    deal_test_dataset=TensorDataset(test_data,test_label)
    train_loader=DataLoader(dataset=deal_train_dataset,batch_size=Batch_size,shuffle=True)
    test_loader=DataLoader(dataset=deal_test_dataset,batch_size=Batch_size,shuffle=True)
    if (i==67):
        model = SimpleNet3()
        model=torch.load('modelv1_'+str(i)+'.pkl')
        if(torch.cuda.is_available()):
            device = torch.device("cuda:0")
            model=model.to(device)
            print('1111111')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.8)
    NUM_EPOCHS=1
    print('beginning training..')
    losslist2=[] 
    acclist=[]  
    for epoch in range(NUM_EPOCHS):
        losslist=[]
        cnt=0
        running_loss = 0.0
        for boards, label in train_loader:
            if(torch.cuda.is_available()):
                device = torch.device("cuda:0")
                model=model.to(device)
                boards=boards.double().cuda()
                label=label.double().cuda()
            label=label.long()
            optimizer.zero_grad()
            outputs = model(boards)
            # loss
            loss = criterion(outputs, label)
            # backward
            loss.backward()
            # update weights
            optimizer.step()
            # print statistics
            running_loss += loss.data
            print(str(cnt)+' epoch:%d, loss: %.3f' %(epoch + 1, running_loss ))
            losslist.append(running_loss)
            running_loss = 0.0
            cnt=cnt+1
        print(sum(losslist)/len(losslist))
        losslist2.append(sum(losslist)/len(losslist))
        

    print("Finished Training")
    modelsave=model.cpu()
    torch.save(modelsave.double(),'onlinemodel.pkl')
    if(i%8==0):
        modelsave=model.cpu()
        torch.save(modelsave.double(),'modelv1_'+str(i+1)+'.pkl')
    
    #********************************************************************
    af=0
    with open('resulttest.csv', 'w', newline='') as csv_file:
        af=af+1
    scores=[]
    N_TESTS=160
    yboard=np.empty(shape=[0,18])
    for i2 in range(N_TESTS):
        gametest = Game(4, score_to_win=2048, random=False)
        
        agent = DIY21Agent(gametest,version=i+1,filenum=i+1)
        xboard=agent.play(verbose=True)
        yboard=np.vstack((yboard,xboard))
        s=gametest.score
        scores.append(s)
    countelem={}
    for itemc in set(scores):
        countelem[itemc]=scores.count(itemc)  
    #rate=countelem[64]/len(scores)
    #if(countelem[64]<rate_history[-1]):
        #rate_history.append(countelem[64])
        #nicenet.append(i+1)
        #torch.save(model,'nicemodelv1_'+str(i+1)+'.pkl')  
    f=open('myresult3_2.txt','a')
    f.write('\n')
    f.write('*********************'+str(i)+'******************************************')
    f.write('\n')
    for ss in scores:
        f.write(str(ss))
        f.write(', ')
    f.write('\n')
    f.write('[')
    for cc in countelem:
        f.write(str(cc)+': '+str(countelem[cc]))
        f.write(', ')
    f.write(']')
    f.write('\n')
    f.write('rate_history:[')
    for rr in rate_history:
        f.write(str(rr)+', ')
    f.write(']')
    f.write('\n')
    f.write('nicenet:[')
    for nice in nicenet:
        f.write(str(nice)+', ')
    f.write(']')
    f.write('\n')
    f.write('Average scores: @'+str(N_TESTS)+'times:'+ str(sum(scores)/len(scores) ))
    f.close()

    #*******************************************************************
    '''
    af=0
    with open('resulttest.csv', 'w', newline='') as csv_file:
        af=af+1
    N_TESTS=100
    for i2 in range(N_TESTS):
        gametest = Game(4, score_to_win=2048, random=False)
        agent = DIY2Agent(gametest,version=i+1)
        agent.play(verbose=True)
    diyend=np.loadtxt('resulttest.csv',usecols=(0,1,2,3,4,5),delimiter=",")
    endindx=len(diyend)
    
    for i3 in range(8):
        gameeasy = Game(4, score_to_win=256, random=False)
        agente = ExpectiMaxAgent(gameeasy)
        agente.play(verbose=True)
   '''

    
    #*********************************************************************** 
    
    

    #*********************************************************************
    
    #boardsets=np.loadtxt('b1024-'+str(i+1)+'.csv',usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),delimiter=',')
    #boardsets=csv.reader(open('resulttest.csv'))
    #for row in boardsets:
        #print row
    boardsets=yboard
    rest=np.empty(shape=[100000,17])
    i3=0
    print(1)
    for item2 in boardsets:
        tmpitem2=item2[0:16]
        boardtmp=tmpitem2.reshape(4,4)
        direction = board_to_move(boardtmp)
        item2tmp=np.append(tmpitem2,[direction],axis=0)
        rest[i3]=item2tmp
        i3=i3+1
    saveboard=rest[0:i3]
    #with open('b2048-'+str(i+1)+'.csv', 'a', newline='') as csv_file:
        #csv_writer = csv.writer(csv_file)
        #for iw in range(i3):
            #csv_writer.writerow(rest[iw])
        
        
    


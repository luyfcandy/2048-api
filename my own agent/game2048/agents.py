import numpy as np
import csv
import torch
class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None,filenum=-1):
        self.game = game
        self.display = display
        self.filenum = filenum

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        result_capacity=np.empty(shape=[2000,18])
        while (n_iter < max_iter) and (not self.game.end):
            result=[]
            
            for i in range(4):
                for j in range(4):
                    result.append(self.game.board[i][j])
            direction = self.step()
            self.game.move(direction)
            
            result.append(direction)
            result.append(self.game.score)
            result_capacity[n_iter]=np.array(result)
            n_iter += 1
                
            if verbose:
                #print("Iter: {}".format(n_iter))
                #print("======Direction: {}======".format(
                    #["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)
        return result_capacity[0:n_iter]
        #if(self.filenum==-1):
            #with open('resulttest.csv', 'a', newline='') as csv_file:
                #csv_writer = csv.writer(csv_file)
                #for i in range(n_iter):
                    #csv_writer.writerow(result_capacity[i])
        #elif(self.filenum>-1):
            #with open('b1024-'+str(self.filenum)+'.csv', 'a', newline='') as csv_file:
                #csv_writer = csv.writer(csv_file)
                #for i2 in range(n_iter):
                    #csv_writer.writerow(result_capacity[i2])
    
    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self,filenum=-1):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None,filenum=-1):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display,filenum)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


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

class DIYAgent(Agent):
    def __init__(self, game, display=None,version=9,filenum=-1):
        super().__init__(game, display,filenum)
        self.model0=torch.load('nicemodelv0_32.pkl')
        self.version=version
        
        self.model1=torch.load('modelv1_'+str(self.version)+'.pkl')
        #self.model2=torch.load('modelv2_1_2.pkl')
        self.modeldict={2:self.model0,4:self.model0,8:self.model0,16:self.model0,
                    32:self.model0,64:self.model1,128:self.model1,256:self.model1,512:self.model1,1024:self.model1}
        
    def step(self):
        boardlst=[]
        for i in range(4):
            for j in range(4):
                boardlst.append(self.game.board[i][j])
        p=np.array(boardlst)
  
        a=grid_ohe(p)
        a=a.astype(np.double)
        torch.set_default_tensor_type('torch.DoubleTensor')
        a0=torch.from_numpy(a)
        a0.unsqueeze_(0)
        s=self.game.score
        rmodel=self.modeldict[s]
        output1=rmodel(a0)
        _, predicted2 = torch.max(output1.data, 1)
        p=predicted2.numpy()
        return p[0]
                    
class DIY2Agent(Agent):
    def __init__(self, game, display=None,version=68,filenum=-1):
        super().__init__(game, display,filenum)
        self.version=version
        
        self.model1=torch.load('modelv1_'+str(self.version)+'.pkl')
        
    def step(self):
        boardlst=[]
        for i in range(4):
            for j in range(4):
                boardlst.append(self.game.board[i][j])
        p=np.array(boardlst)
  
        a=grid_ohe(p)
        a=a.astype(np.double)
        torch.set_default_tensor_type('torch.DoubleTensor')
        a0=torch.from_numpy(a)
        a0.unsqueeze_(0)
        output1=self.model1(a0)
        #print(output1)
        _, predicted2 = torch.max(output1.data, 1)
        p=predicted2.numpy()
        #print(p)
        return p[0]
		
class DIY21Agent(Agent):
    def __init__(self, game, display=None,version=68,filenum=-1):
        super().__init__(game, display,filenum)
        self.version=version
        
        self.model1=torch.load('onlinemodel.pkl')
        
    def step(self):
        boardlst=[]
        for i in range(4):
            for j in range(4):
                boardlst.append(self.game.board[i][j])
        p=np.array(boardlst)
  
        a=grid_ohe(p)
        a=a.astype(np.double)
        torch.set_default_tensor_type('torch.DoubleTensor')
        a0=torch.from_numpy(a)
        a0.unsqueeze_(0)
        output1=self.model1(a0)
        #print(output1)
        _, predicted2 = torch.max(output1.data, 1)
        p=predicted2.numpy()
        #print(p)
        return p[0]

class DIY3Agent(Agent):
    def __init__(self, game, display=None,version=0,filenum=-1):
        super().__init__(game, display,filenum)
        self.modela=torch.load('modelv1_26.pkl')
        self.modelb=torch.load('modelv1_30.pkl')
        self.modelc=torch.load('modelv1_31.pkl')
        self.model2a=torch.load('modelv1_'+str(version)+'.pkl')

    def step(self):
        boardlst=[]
        for i in range(4):
            for j in range(4):
                boardlst.append(self.game.board[i][j])
        p=np.array(boardlst)
  
        a=grid_ohe(p)
        a=a.astype(np.double)
        torch.set_default_tensor_type('torch.DoubleTensor')
        a0=torch.from_numpy(a)
        a0.unsqueeze_(0)
        s=self.game.score
        if(s<=32):
            output1=self.modela(a0)
            output2=self.modelb(a0)
            output3=self.modelc(a0)
        
            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1)
            _, predicted3 = torch.max(output3.data, 1)
            p1=predicted1.numpy()[0]
            p2=predicted2.numpy()[0]
            p3=predicted3.numpy()[0]
            if(p1==p2 and p2!=p3):
                dec=p1
            else:
                dec=p3
        if(s>=64):
            outputb1=self.modela(a0)
            _, predictedb1 = torch.max(outputb1.data, 1)
            pb1=predictedb1.numpy()[0]
            dec=pb1
        return dec

class DIY4Agent(Agent):
    def __init__(self, game, display=None,version=0,filenum=-1):
        super().__init__(game, display,filenum)
        self.version=version
        
        self.modela=torch.load('modelv1_26.pkl')
        self.modelb=torch.load('modelv1_30.pkl')
        self.modelc=torch.load('modelv1_31.pkl')
        self.modeld=torch.load('modelv1_11.pkl')
        self.modele=torch.load('modelv1_12.pkl')
        
    def step(self):
        boardlst=[]
        for i in range(4):
            for j in range(4):
                boardlst.append(self.game.board[i][j])
        p=np.array(boardlst)
  
        a=grid_ohe(p)
        a=a.astype(np.double)
        torch.set_default_tensor_type('torch.DoubleTensor')
        a0=torch.from_numpy(a)
        a0.unsqueeze_(0)
        output1=self.modela(a0)
        output2=self.modelb(a0)
        output3=self.modelc(a0)
        output4=self.modeld(a0)
        output5=self.modele(a0)
        outputall=output1+output2+output3+output4+output5
        print(outputall)
        _, predicted2 = torch.max(outputall.data, 1)
        p=predicted2.numpy()
        print(p)
        return p[0]



















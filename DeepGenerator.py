
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:46:32 2020

@author: Abhilash
"""
#trying for sequential multi-layered
import numpy as np
import matplotlib.pyplot as plt

class DeepGenerator:

    def sigmoid(Z):
        return (1/1+np.exp(-Z))

    def dsigmoid(Z):
        return (Z*(1-Z))

    def dtanh(Z):
        return (1.0- Z**2) 


    def data_abstract(self,path,choice):
        data=open(path,'r').read()
        if(choice=='character_generator'):
            data=data
        elif(choice=='word_generator'):
            data=data.split()
        return data
    
    def data_preprocess(self,path,choice):
        data=DeepGenerator.data_abstract(self,path,choice)
        chars=list(set(data))
        data_size,vocab_size=len(data),len(chars)
        char_to_idx=dict((char,idx) for idx,char in enumerate(chars))
        idx_to_char=dict((idx,char) for idx,char in enumerate(chars))
        return data,data_size,vocab_size,char_to_idx,idx_to_char

    def hyperparamteres(self,hidden_layers_size,no_hidden_layers,learning_rate,step_size,vocab_size):
        #hyperparameters
        if(step_size>vocab_size):
            raise ValueError('Step Size must be less than vocabulary size')
        hidden_layers=hidden_layers_size
        learning_rate=learning_rate
        step_size=step_size
        hid_layer=no_hidden_layers
        #weights and biases
        Wxh=np.random.randn(hidden_layers,vocab_size)*0.05
        Whh1=np.random.randn(hidden_layers,hidden_layers)*0.05
        Whh_vector=[]
        for i in range(0,hid_layer):
            Whh_vector.append(np.random.randn(hidden_layers,hidden_layers)*0.05)
    
        Whh=np.random.randn(hidden_layers,hidden_layers)*0.05
        Why=np.random.randn(vocab_size,hidden_layers)*0.05
        bh1=np.zeros((hidden_layers,1))
        bh=np.zeros((hidden_layers,1))
        bh_vector=[]
        for i in range(0,hid_layer):
            bh_vector.append(np.zeros((hidden_layers,1)))
        by=np.zeros((vocab_size,1))
        #losses=[]    
        return hidden_layers,learning_rate,step_size,hid_layer,Wxh,Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by
        


    def loss_evaluation(self,inp,target,h_previous,hidden_layers,hid_layer,Wxh,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by):
    
        x_state,y_state,h_state,h1_state,p_state={},{},{},{},{}
        #copy previous state
        h_vector=[]
        for i in range(0,hid_layer):
            h_vector.append({})
        
        h1_state[-1]=np.copy(h_previous)
        h_state=h1_state
        for i in range(0,hid_layer):
            h_vector[i]=h_state
        loss=0
        for i in range(len(inp)):
            #encode
            x_state[i]=np.zeros((vocab_size,1))
            x_state[i][inp[i]]=1
            #forward pass
            h1_state[i]=np.tanh(np.dot(Wxh,x_state[i]) + np.dot(Whh1,h1_state[i-1]) + bh1)
            h_vector[0][i]=np.tanh(np.dot(Whh1,h1_state[i])+ np.dot(Whh_vector[0],h_vector[0][i-1]) + bh_vector[0])
            #sequential looping
            for j in range(1,hid_layer-1):
                h_vector[j][i]=np.tanh(np.dot(Whh_vector[j-1],h_vector[j-1][i]) + np.dot(Whh_vector[j],h_vector[j][i-1]) + bh_vector[j])
            h_vector[hid_layer-1][i]=np.tanh(np.dot(Whh_vector[hid_layer-2],h_vector[hid_layer-2][i]) + np.dot(Whh_vector[hid_layer-1],h_vector[hid_layer-1][i-1]) + bh_vector[hid_layer-1])
            h_state[i]=np.tanh(np.dot(Whh_vector[hid_layer-1],h_vector[hid_layer-1][i])+ np.dot(Whh,h_state[i-1]) + bh)
            y_state[i]=np.dot(Why,h_state[i])+by
            p_state[i]=np.exp(y_state[i])/np.sum(np.exp(y_state[i]))
            #crossentropy
            loss+=-np.log(p_state[i][target[i],0])
        #initialise for derivatives    
        dWxh,dWhh1,dWhh,dWhy=np.zeros_like(Wxh),np.zeros_like(Whh1),np.zeros_like(Whh),np.zeros_like(Why)
        dWhh_vector=[]
        for j in range(0,hid_layer):
            dWhh_vector.append(np.zeros_like(Whh_vector[j]))
        dbh1,dbh,dby=np.zeros_like(bh1),np.zeros_like(bh),np.zeros_like(by)
        dbh_vector=[]
        for j in range(0,hid_layer):
            dbh_vector.append(np.zeros_like(bh_vector[j]))
        dhnext=np.zeros_like(h_state[0])
        dh1next=np.zeros_like(h1_state[0])
        dhnext_vector=[]
        for j in range(0,hid_layer):
            dhnext_vector.append(np.zeros_like(h_vector[j][0]))
        dh_vector=[]
        for j in range(0,hid_layer):
            dh_vector.append(np.zeros_like(h_vector[j][0]))
        for i in reversed(range(len(inp))):
        
            #backward pass
            dy=np.copy(p_state[i])
            dy[target[i]]-=1
            #(d(y*h_state_i.T)/d(Why)) ->derivative wrt y
            dWhy +=np.dot(dy,h_state[i].T)
            #change bias
            dby+=dy
            #d(Why.T * y)/d(h)->derivative wrt h
            #add gradients
            dh= np.dot(Why.T,dy) + dhnext
            #end of last hidden layer
        
        
            #intermediate hidden layer 
            #d(tanh(h1_state_i))/d(h)->derivative of tanh
            #chain rule derivative through tanh layer
            #d(tanh(h1_state_i))/d(Wxh)
            #-> d(d(tanh(h1_state_i)/d(h)))/d(Whh)
            #->d(dhr*(h1_state_i))/d(Whh)
            #->d(dhr*(h1_state_i))/d(h1_state_i)*(d(h1_state_i)/d(Whh))
        
            dhr= (1-h_state[i]**2)*dh
            dbh+=dhr
            #d(dhr*h1_state_i-1)/d(Whh)->derivative wrt Whh
            dWhh+=np.dot(dhr,h_state[i-1].T)
            #d(dhr*h1_state_i)/d(Whh1)->derivative wrt Whh1
            dWhh_vector[hid_layer-1]+=np.dot(dhr,h_vector[hid_layer-1][i].T)
            #add gradients
            #d(Whh1.T*dhr)/d(h1)->derivative wrt h1
            dh_vector[hid_layer-1]= np.dot(Whh_vector[hid_layer-1].T,dhr) + dhnext_vector[hid_layer-1]
            #d(Whh.T*dhr)/d(h)->derivative wrt h
            dh= np.dot(Whh.T,dhr) + dhnext
            dhnext=np.dot(Whh.T,dhr)
            dhnext_vector[hid_layer-1]=np.dot(Whh_vector[hid_layer-1].T,dhr)
            #end of last intermediate hidden
        
            #merge between last hidden with the end of sequential
            dhr7=(1-h_vector[hid_layer-1][i]**2)*dh
            dbh_vector[hid_layer-1]+=dhr7
            #d(dhr7*h_state_i)/d(Whh)->derivative wrt Whh
            dWhh+=np.dot(dhr7,h_state[i-1].T)
            #d(dhr7*h7_state_i)/d(Whh7)->derivative wrt Whh7
            dWhh_vector[hid_layer-1]+=np.dot(dhr7,h_vector[hid_layer-1][i].T)
            #add gradients
            #d(Whh7.T*dhr7)/d(h7)->derivative wrt h7
            dh_vector[hid_layer-1]=np.dot(Whh_vector[hid_layer-1].T,dhr7) + dhnext_vector[hid_layer-1]
            #d(Whh.T*dhr)/d(h)->derivative wrt h
            dh=np.dot(Whh.T,dhr7)+ dhnext
            dhnext=np.dot(Whh.T,dhr7)
            dhnext_vector[hid_layer-1]=np.dot(Whh_vector[hid_layer-1].T,dhr7)
        
            #inside sequential backprop loop
            for j in range(hid_layer-2,0,-1):
                #dtanh of each vector
                dhj=(1-h_vector[j][i]**2)*dh_vector[j+1]
                #biass add of each layer
                dbh_vector[j]+=dhj
                #prev_layer from end backprop->7->6->5->4...
                #d(dh_eachlayer*h_prev_layer(j+1)_i-1)/dWhh_prev_layer
                dWhh_vector[j+1]+=np.dot(dhj,h_vector[j+1][i-1].T)
                #d(dh_eachlayer*h_currentlayer_i)/dWhh_current_layer
                dWhh_vector[j]+= np.dot(dhj,h_vector[j][i].T)
                #d(Whh_previous_layer*dhj_current_layer)d(h_current) + addgradient_prev_layer
                #derivative wrt h_prev
                dh_vector[j+1]=np.dot(Whh_vector[j+1].T,dhj) + dhnext_vector[j+1]
                #d(Whh_current_layer*dhj_current_layer)d(h_current) + addgradient_current_layer
                #derivative wrt h_current
                dh_vector[j]=np.dot(Whh_vector[j].T,dhj) + dhnext_vector[j]
                #update next_vector_prev
                dhnext_vector[j+1]=np.dot(Whh_vector[j+1].T,dhj)
                #update next_vector_current
                dhnext_vector[j]=np.dot(Whh_vector[j].T,dhj)
            
            #merge between first input with the first layer of sequential    
            #first hidden layer 
            #d(tanh(h_state_i))/d(h)->derivative of tanh
            #chain rule derivative through tanh layer
            #d(tanh(h_state_i))/d(Wxh)
            #-> d(d(tanh(h_state_i)/d(h)))/d(Whh1)
            #->d(dhr*(h_state_i))/d(Whh1)
            #->d(dhr*(h_state_i))/d(h_state_i)*(d(h_state_i)/d(Whh1))
        
            dhr1=(1-h_vector[0][i]**2)*dh_vector[1]
            dbh1+=dhr1
            #d(dhr1*h_prev_layer(0)_i-1)/dWhh_prev_layer[0]
            dWhh_vector[0]+=np.dot(dhr1,h_vector[0][i-1].T)
            #d(dhr1*h1_state_i)/dWhh1
            dWhh1= np.dot(dhj,h1_state[i].T)
            #next h d(Whh1,dhr1)/d(hnext)->derivative wrt h1_state
            dh1next= np.dot(Whh1.T,dhr1)
            dhnext_vector[0]=np.dot(Whh_vector[0].T,dhr1)
        
        return loss,dWxh,dWhh1,dWhh_vector,dWhh,dWhy,dbh1,dbh_vector,dbh,dby,h_state[len(inp)-1],Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by


    def output_sample(h1,seed_ix,n,vocab_size,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by,hid_layer):
        #return max probability of the chars
        #normal forward pass 
        x=np.zeros((vocab_size,1))
        x[seed_ix]=1
        ixs=[]
        h=h1
        h_vector=[]
        for i in range(0,hid_layer):
            h_vector.append(h)
        for i in range(n):
            h1=np.tanh(np.dot(Wxh,x)+np.dot(Whh1,h1)+bh1)
            for j in range(0,hid_layer-1):
                h_vector[j]=np.tanh(np.dot(Whh_vector[j-1],h_vector[j-1]) + np.dot(Whh_vector[j],h_vector[j]) + bh_vector[j])
            h_vector[hid_layer-1]=np.tanh(np.dot(Whh_vector[hid_layer-2],h_vector[hid_layer-2]) + np.dot(Whh_vector[hid_layer-1],h_vector[hid_layer-1]) + bh_vector[j])
            h=np.tanh(np.dot(Whh_vector[hid_layer-1],h_vector[hid_layer-1])+np.dot(Whh,h)+bh)
            y=np.dot(Why,h)+by
            p=np.exp(y)/np.sum(np.exp(y))
            ix=np.random.choice(range(vocab_size),p=p.ravel())
            x=np.zeros((vocab_size,1))
            x[ix]=1
            ixs.append(ix)
        return ixs

    def plot_gradient(self,gradient_analytical,error):
        plt.plot(gradient_analytical,error)
        plt.xlabel("Gradients Analytical")
        plt.ylabel("Error")
        plt.show()
    
    def plot_loss(self,epochs,gradient_loss):
        plt.plot(epoch,gradient_loss)
        plt.ylabel("Gradients Loss")
        plt.xlabel("Epochs")
        plt.show()
    
        
    def start_predict(self,count,epochs,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by,hid_layer,char_to_idx,idx_to_char,vocab_size,learning_rate,step_size,data,hidden_layers):
        n,p=0,0
        gradient_loss=[]
        epoch=[]
        hidden_layers,learning_rate,step_size,hid_layer,Wxh,Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by=DeepGenerator.hyperparamteres(self,hidden_layers_size,no_hidden_layers,learning_rate,step_size,vocab_size)
        #adagrad memory storage
        mWhh_vector=[]
        for j in range(0,hid_layer):
            mWhh_vector.append(np.zeros_like(Whh_vector[j]))
        mWxh,mWhh1,mWhh,mWhy=np.zeros_like(Wxh),np.zeros_like(Whh1),np.zeros_like(Whh),np.zeros_like(Why)
        mbh_vector=[]
        for j in range(0,hid_layer):
            mbh_vector.append(np.zeros_like(bh_vector[j]))
        mbh1,mbh,mby=np.zeros_like(bh1),np.zeros_like(bh),np.zeros_like(by)
        tau=1e-8
        hypo_loss= -np.log(1.0/vocab_size)*step_size
        out_text=""
        #looping for training
        for j in range(0,epochs):
    
            if p+step_size+1>= len(data) or n==0:
                #initial step
                h_prev=np.zeros((hidden_layers,1))
                p=0
    
            inp=[char_to_idx[c] for c in data[p:p+step_size]]
            target=[char_to_idx[c] for c in data[p+1:p+step_size+1]]
    
    
            loss,dWxh,dWhh1,dWhh_vector,dWhh,dWhy,dbh1,dbh_vector,dbh,dby,h_prev,Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by=DeepGenerator.loss_evaluation(self,inp,target,h_prev,hidden_layers,hid_layer,Wxh,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by)
            hypo_loss=hypo_loss*0.999 + loss*0.001
            if n%100==0:
                sample_ixs=DeepGenerator.output_sample(h_prev,inp[0],count,vocab_size,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by,hid_layer)
                print("generated text:")
                out_txt=' '.join(idx_to_char[i] for i in (sample_ixs))
                print("==========================")
                print(out_txt,)
                print("=========================")

            if n%50==0 and n>0:
                print("Epoch: ",n," Loss: ",hypo_loss)
                print("=====>")
                epoch.append(n)
                gradient_loss.append(hypo_loss)
                #losses.append(hypo_loss)
            Whh_vector_sum=np.sum(Whh_vector)
            bh_vector_sum=np.sum(bh_vector)
            dWhh_vector_sum=np.sum(dWhh_vector)
            dbh_vector_sum=np.sum(dbh_vector)
            mWhh_vector_sum=np.sum(mWhh_vector)
            mbh_vector_sum=np.sum(mbh_vector)
            for param,dparam,mem in zip([Wxh,Whh1,Whh_vector_sum,Whh,Why,bh,bh_vector_sum,bh,by],[dWxh,dWhh1,dWhh_vector_sum,dWhh,dWhy,dbh1,dbh_vector_sum,dbh,dby],[mWxh,mWhh1,mWhh_vector_sum,mWhh,mWhy,mbh1,mbh_vector_sum,mbh,mby]):
                mem+=dparam**2
                param+=- learning_rate*dparam/np.sqrt(mem+tau)
                dparam_length=dparam.shape
                param_length=param.shape
                if(dparam_length != param_length):
                        raise  ValueError('Error dimensions dont match:dparam and param')   
            p+=step_size
            n+=1
        return epoch,gradient_loss
    
if __name__=='__main__':
    deepgen=DeepGenerator()
    learning_rate=1e-1
    step_size=25
    no_hidden_layers=24
    hidden_layers_size=64
    path='C:\\Users\\User\\Desktop\\test2.txt'
    choice='word_generator'
    epochs=40000
    count=100
    data,data_size,vocab_size,char_to_idx,idx_to_char=deepgen.data_preprocess(path,choice)
    hidden_layers,learning_rate,step_size,hid_layer,Wxh,Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by=deepgen.hyperparamteres(hidden_layers_size,no_hidden_layers,learning_rate,step_size,vocab_size)
    epoch,gradient_loss=deepgen.start_predict(count,epochs,Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by,hid_layer,char_to_idx,idx_to_char,vocab_size,learning_rate,step_size,data,hidden_layers)
    print(gradient_loss)
    deepgen.plot_loss(epoch,gradient_loss)
    






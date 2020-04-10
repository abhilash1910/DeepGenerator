# DeepGenerator
Sentence Sequence Transduction Library (Seq to Seq) for text generation using sequential generative Vanilla RNN using numpy  
The library is a generative network built purely in numpy and matplotlib. The library is a sequence to sequence transduction
library for generating n sequence texts from given corpus. Metrics for accuracy-BLEU has provided a considerable accuracy metric
for epochs greater than 20000.The library is sequential and includes intermediate tanh activation in the intermediate stages with 
softmax cross entropy loss ,and generalised Adagrad optimizer.

# library facts:
  
    initialisation:  
        
        import DeepGenerator.DeepGenerator as dg
    ====================
    
    creating object:
        deepgen=dg.DeepGenerator()
    
# Functions: 
     1.attributes for users- learning rate,epochs,local path of data storage(text format),number of hidden layers,kernel size,sequence/step size,count of next words
      2.data_abstract function- Takes arguements (self,path,choice) - 
                                path= local path of text file
                                choice= 'character_generator' for character generation network
                                        'word_generator' for word generator network
                                Returns data
                                Usage- ouput_data=deepgen.data_preprocess(DeepGenerator.path,DeepGenerator.choice)
      3.data_preprocess function- Takes arguements (self,path,choice)-
                                path= local path of text file
                                choice= 'character_generator' for character generation network
                                        'word_generator' for word generator network
                                Returns data,data_size,vocab_size,char_to_idx,idx_to_char
                                Usage- data,data_size,vocab_size,char_to_idx,idx_to_char=deepgen.data_preprocess(DeepGenerator.path,DeepGenerator.choice)
      4.hyperparameters function-Takes arguements (self,hidden_layers_size,no_hidden_layers,learning_rate,step_size,vocab_size)-
                                hidden_layers-kernel size-recommended under 2048
                                no_hidden_layers- sequential intermediate layers
                                learning_rate- learning_rate (range of 1e-3)
                                step_size- sequence length(should be <= vocab_size)
                                vocab_size
                                Returns hidden_layers,learning_rate,step_size,hid_layer,Wxh,Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by
                                Usage- hidden_layers,learning_rate,step_size,hid_layer,Wxh,Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by=deepgen.hyperparamteres(dg.hidden_layers_size,dg.no_hidden_layers,dg.learning_rate,dg.step_size,dg.vocab_size)
      5. loss_evaluation function- Takes arguements    (self,inp,target,h_previous,hidden_layers,hid_layer,Wxh,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by) -
                                inp= character to indices encoded dictionary of input text
                                target=character to indices encoded dictionary of generated text
                                h_previous-value of hidden layer for previous state
                                hidden_layers-kernel size
                                hid_layer-sequential hidden layers
                                ---------- sequential layers---------
                                       -----weight tensors------
                                Wxh- weight tensor of input to first hidden layer
                                Wh1- weight tensor of first hidden layer to first layer of sequential network
                                Whh_vector-weight tensors of intermediate sequential network
                                Whh- weight tensor of last sequential to last hidden layer
                                Why-weight tensor of last hidden layer to output layer
                                        -----bias tensors-------
                                bh1-bias of first hidden layer
                                bh_vector-bias of intermediate sequential layers
                                bhh-bias of end hidden layer
                                by-bias at output
                                
                                Returns loss,dWxh,dWhh1,dWhh_vector,dWhh,dWhy,dbh1,dbh_vector,dbh,dby,h_state[len(inp)-1],Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by
                                Usage loss,dWxh,dWhh1,dWhh_vector,dWhh,dWhy,dbh1,dbh_vector,dbh,dby,h_state[len(inp)-1],Whh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by=deepgen.loss_evaluation(dg.inp,dg.target,dg.h_previous,dg.hidden_layers,dg.hid_layer,dg.Wxh,dg.Wh1,dg.Whh_vector,dg.Whh,dg.Why,dg.bh1,dg.bh_vector,dg.bh,dg.by)
      6.start_predict function-Takes arguements (self,count,epochs,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by,hid_layer,char_to_idx,idx_to_char,vocab_size,learning_rate,step_size,data,hidden_layers)
                                counts-count of sequences to generate
                                epochs-epochs
                                Whi -weight tensors
                                bhi-bias tensors
                                hid_layer-no of sequential layers
                                char_to_idx-character to index encoder
                                idx_to_char-index to character decoder
                                vocab_size-vocab_size
                                learning_rate-learning_rate
                                step_size-sequence length
                                hidden_layers-kernel size
                                Returns epochs and gradient losses vector
                                Usage-epochs,gradient_loss=deepgen.start_predict(dg.count,dg.epochs,dg.Whh1,dg.Whh_vector,dg.Whh,dg.Why,dg.bh1,dg.bh_vector,dg.bh,dg.by,dg.hid_layer,dg.char_to_idx,dg.idx_to_char,dg.vocab_size,dg.learning_rate,dg.step_size,dg.data,dg.hidden_layers) 
       7.output_sample function- Takes arguements (self,h1,seed_ix,n,vocab_size,Wh1,Whh_vector,Whh,Why,bh1,bh_vector,bh,by,hid_layer)-
                                h1-hidden layer previous state
                                seed_ix-starting point for generation
                                n-count of text to generate
                                Whi-weight tensor
                                bhi-bias tensor
                                hid_layer-no of sequential layers
                                Returns ixs- integer vector of maximum probability characters/words
                                Usage-ixs=deepgen.output_sample(dg.h1,dg.seed_ix,dg.n,dg.vocab_size,dg.Wh1,dg.Whh_vector,dg.Whh,dg.Why,dg.bh1,dg.bh_vector,dg.bh,dg.by,dg.hid_layer)
       8.plot_loss function  -Takes arguements(self,epochs,gradient_loss)-
                              epochs-epoch vector
                              gradient_loss- gradient loss vector
                              Returns void
                              Usage-deepgen.plot_loss(dg.epoch,dg.gradient_loss)
# Usage-
The file sample.py contains the usage specification and syntax for generating text
Jupyter notebook -Deepgen.ipynb is also present as a sample with different text files.

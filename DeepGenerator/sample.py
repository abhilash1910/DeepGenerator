# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:17:00 2020
@author: User
"""

import DeepGenerator.DeepGenerator as dg



if __name__=='__main__':
    deepgen=dg.DeepGenerator()
    dg.learning_rate=1e-1
    dg.step_size=25
    dg.no_hidden_layers=24
    dg.hidden_layers_size=64
    dg.path='C:\\Users\\User\\Desktop\\test2.txt'
    dg.choice='word_generator'
    dg.epochs=100
    dg.count=100
    
    dg.data,dg.data_size,dg.vocab_size,dg.char_to_idx,dg.idx_to_char=deepgen.data_preprocess(dg.path,dg.choice)
    print(dg.vocab_size)
    dg.hidden_layers,dg.learning_rate,dg.step_size,dg.hid_layer,dg.Wxh,dg.Whh1,dg.Whh_vector,dg.Whh,dg.Why,dg.bh1,dg.bh_vector,dg.bh,dg.by=deepgen.hyperparamteres(dg.hidden_layers_size,dg.no_hidden_layers,dg.learning_rate,dg.step_size,dg.vocab_size)
    dg.epoch,dg.gradient_loss,dg.out_txt_vector=deepgen.start_predict(dg.count,dg.epochs,dg.Whh1,dg.Whh_vector,dg.Whh,dg.Why,dg.bh1,dg.bh_vector,dg.bh,dg.by,dg.hid_layer,dg.char_to_idx,dg.idx_to_char,dg.vocab_size,dg.learning_rate,dg.step_size,dg.data,dg.hidden_layers)
    print(dg.gradient_loss)
    print(dg.out_txt_vector)
    deepgen.plot_loss(dg.epoch,dg.gradient_loss)
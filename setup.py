# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:45:49 2020

@author: Abhilash
"""

from distutils.core import setup
setup(
  name = 'DeepGenerator',         
  packages = ['DeepGenerator'],   
  version = '0.1',       
  license='MIT',        
  description = 'Sentence Sequence Transduction Library (Seq to Seq) for text generation using sequential generative Vanilla RNN using numpy.',   
  author = 'ABHILASH MAJUMDER',
  author_email = 'abhilash.majumder@hsbc.co.in',
  url = 'https://github.com/abhilash1910/DeepGenerator',   
  download_url = 'https://github.com/abhilash1910/DeepGenerator/archive/v_01.tar.gz',    
  keywords = ['sequence_to_sequence', 'text generation', 'sentence generation','generative network','RNN','Vanilla RNN','generation library'],   
  install_requires=[           

          'numpy',         
          'matplotlib' 
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
     'Programming Language :: Python :: 2.5',      
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

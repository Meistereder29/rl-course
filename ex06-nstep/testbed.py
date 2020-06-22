# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:57:31 2020

@author: Gregor
"""


def outerFunction():
    
    a = 10
    print("Outer Function: " +str(a))
    def innerFunction():
        a = 2
        print("Inner Function: " +str(a))
    innerFunction()
    print("Outer Function 2: " +str(a))

outerFunction()

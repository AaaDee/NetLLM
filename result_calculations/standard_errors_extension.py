#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Calculates the standard errors for the comparison of estimated means computed for the
# For simplicity and clarity, the source values are hard-coded into the script

import math

# sample 1: llama2 result, sample 2: llama3 results
# m1, m2: means
# s1, s2 standard deviations
# n1, n2: sample size 
def calculate_t(m1, m2, s1, s2, n1, n2):
    return (m1 - m2) / math.sqrt((s1**2/n1) + (s2**2/n2))



# ABR
m1 = 0.7471800397150192
s1 = 1.2414664713891124
n1 = 4700
m2 = 0.7585952097355776
s2 = 1.535666977278166
n2 = 4700

print(calculate_t(m1, m2, s1, s2, n1, n2))


# CJS
m1 = 45.51097027512611
s1 = 42.66186557935327
n1 = 200
m2 = 53.917269227193735
s2 = 46.544997706991964
n2 = 200

print(calculate_t(m1, m2, s1, s2, n1, n2))


# VP
m1 = 11.513970512104786
s1 = 6.585152671755119
n = 127
m2 = 12.447741389695832
s2 = 11.498545942483883
n2 = 1698

print(calculate_t(m1, m2, s1, s2, n1, n2))


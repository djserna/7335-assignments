# -*- coding: utf-8 -*-
"""
@author: Daniel Serna
"""

import numpy as np
import numpy.lib.recfunctions as rfn
from collections import OrderedDict
import matplotlib.pyplot as plt

#NumPy Cheatsheet - https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf

## HW notes:
'''    
A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.

     B. How much was paid for J-codes to providers for 'in network' claims?

     C. What are the top five J-codes based on the payment to providers?



2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.

    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.

    B. What insights can you suggest from the graph?

    C. Based on the graph, is the behavior of any of the providers concerning? Explain.



3. Consider all claim lines with a J-code.

     A. What percentage of J-code claim lines were unpaid?

     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

     C. How accurate is your model at predicting unpaid claims?

      D. What data attributes are predominately influencing the rate of non-payment?
'''

#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
#These are the data types or dtypes that will be used in the below function, np.genfromtxt()
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']

#read in the claims data into a structured numpy array
CLAIMS = np.genfromtxt('data\claim.sample.csv', dtype=types, delimiter=',', names=True, 
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])
print(CLAIMS.dtype.names)

#Question 1a
testJCode = 'J'
testJCode = testJCode.encode()

#Try find() on CLAIMS
JcodeIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], testJCode, start=1, end=2)!=-1)

#Using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

print(Jcodes)

print(F"Question 1a: Number of claim lines that have J-codes: {len(Jcodes)}")

#Question 1b
testInNetwork = 'I'
testInNetwork = testInNetwork.encode()
InNetworkJcodeIndexes =np.flatnonzero(np.core.defchararray.find(Jcodes['InOutOfNetwork'], testInNetwork, start=1, end=2)!=-1)
InNetworkJcodes = Jcodes[InNetworkJcodeIndexes]

print(F"Question 1b: Amount paid to in-network providers for J-code claims: {InNetworkJcodes['ProviderPaymentAmount'].sum()}")

#Question 1c
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')

# Reverse the sorted Jcodes (A.K.A. in descending order)
Sorted_Jcodes = Sorted_Jcodes[::-1]

# You can subset it...
ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
Jcodes = Sorted_Jcodes['ProcedureCode']

#Join arrays together
arrays = [Jcodes, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

#GroupBy JCodes using a dictionary
JCode_dict = {}

#Aggregate with Jcodes - code  modifiedfrom a former student's code, Anthony Schrams
for aJCode in Jcodes_with_ProviderPayments:
    if aJCode[0] in JCode_dict.keys():
        JCode_dict[aJCode[0]] += aJCode[1]
    else:
        aJCode[0] not in JCode_dict.keys()
        JCode_dict[aJCode[0]] = aJCode[1]

#sum the JCodes
np.sum([v1 for k1,v1 in JCode_dict.items()])

#create an OrderedDict (which we imported from collections): https://docs.python.org/3.7/library/collections.html#collections.OrderedDict
#Then, sort in descending order
JCodes_PaymentsAgg_descending = OrderedDict(sorted(JCode_dict.items(), key=lambda aJCode: aJCode[1], reverse=True))
top5JCodes = list(JCodes_PaymentsAgg_descending)[:5]
print(F"Question 1c: Top 5 J-codes based on payment to providers: {top5JCodes}")

#Question 2a
unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)

## find paid row indexes
paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)

#Here are our
Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]

Paid_Jcodes = Sorted_Jcodes[paid_mask]

unpaid_dict = {}
paid_dict = {}
combined_dict = {}
for claim in Unpaid_Jcodes:
    if claim['ProviderID'] in unpaid_dict.keys():
        unpaid_dict[claim['ProviderID']] += 1
    else:
        claim['ProviderID'] not in unpaid_dict.keys()
        unpaid_dict[claim['ProviderID']] = 1
        
for claim in Paid_Jcodes:
    if claim['ProviderID'] in paid_dict.keys():
        paid_dict[claim['ProviderID']] += 1
    else:
        claim['ProviderID'] not in paid_dict.keys()
        paid_dict[claim['ProviderID']] = 1
        
for key in (unpaid_dict.keys() | paid_dict.keys()):
    if key in unpaid_dict: combined_dict.setdefault(key, []).append(unpaid_dict[key])
    if key in paid_dict: combined_dict.setdefault(key, []).append(paid_dict[key])

combinedKeys = list(combined_dict.keys())
for key in combinedKeys:
    if len(combined_dict[key]) == 1:
        del combined_dict[key]

providerArray = []
unpaidArray = []
paidArray = []

for key in combined_dict.keys():
    providerArray.append(key)
    unpaidArray.append(combined_dict[key][0])
    paidArray.append(combined_dict[key][1])

plotLabels = providerArray #etc

#Produce the scatterplot as the answer to 2a
fig, ax = plt.subplots()
ax.scatter(unpaidArray, paidArray)
ax.grid(linestyle='-', linewidth='0.75', color='red')

fig = plt.gcf()
fig.set_size_inches(25, 25)
plt.rcParams.update({'font.size': 28})

for i, txt in enumerate(plotLabels):
    ax.annotate(txt, (unpaidArray[i], paidArray[i]))

plt.tick_params(labelsize=35)
plt.xlabel('# of Unpaid claims', fontsize=35)

plt.ylabel('# of Paid claims', fontsize=35)

plt.title('Scatterplot of Unpaid and Paid claims by Provider', fontsize=45)
plt.savefig('Paid_Unpaid_Scatterplot.png')
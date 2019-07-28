# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 18:54:37 2019

@author: User
"""
import numpy as np

# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  Then you should decided if you should split into two groups so eveyone is happier.  

# Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.
  
# This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of decsion making problems that are currently not leveraging machine learning.  



# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  

#I don't understand how we can support the proposed people dictionary structure since there are more properties than the restaurant dictionary.
#This needs to match up with the restaurant dictionary structure, so I will not use all the properties.

#Generate random input values for people.
valueArrayJane = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayJack = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayJill = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayTim = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayBob = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayAndy = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayCharlie = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayPete = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayDaniel = np.array([np.random.dirichlet(np.ones(4),size=1)])
valueArrayBrad = np.array([np.random.dirichlet(np.ones(4),size=1)])

people = {'Jane': {'willingness to travel': valueArrayJane[0,0,0],
                  'desire for new experience': valueArrayJane[0,0,1],
                  'cost': valueArrayJane[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayJane[0,0,3]
                  },
          'Jack': {'willingness to travel': valueArrayJack[0,0,0],
                  'desire for new experience': valueArrayJack[0,0,1],
                  'cost': valueArrayJack[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayJack[0,0,3]
                  },
          'Jill': {'willingness to travel': valueArrayJill[0,0,0],
                  'desire for new experience': valueArrayJill[0,0,1],
                  'cost': valueArrayJill[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayJill[0,0,3]
                  },
          'Tim': {'willingness to travel': valueArrayTim[0,0,0],
                  'desire for new experience': valueArrayTim[0,0,1],
                  'cost': valueArrayTim[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayTim[0,0,3]
                  },
          'Bob': {'willingness to travel': valueArrayBob[0,0,0],
                  'desire for new experience': valueArrayBob[0,0,1],
                  'cost': valueArrayBob[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayBob[0,0,3]
                  },
          'Andy': {'willingness to travel': valueArrayAndy[0,0,0],
                  'desire for new experience': valueArrayAndy[0,0,1],
                  'cost': valueArrayAndy[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayAndy[0,0,3]
                  },
          'Charlie': {'willingness to travel': valueArrayCharlie[0,0,0],
                  'desire for new experience': valueArrayCharlie[0,0,1],
                  'cost': valueArrayCharlie[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayCharlie[0,0,3]
                  },
          'Pete': {'willingness to travel': valueArrayPete[0,0,0],
                  'desire for new experience': valueArrayPete[0,0,1],
                  'cost': valueArrayPete[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayPete[0,0,3]
                  },
          'Daniel': {'willingness to travel': valueArrayDaniel[0,0,0],
                  'desire for new experience': valueArrayDaniel[0,0,1],
                  'cost': valueArrayDaniel[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayDaniel[0,0,3]
                  },
          'Brad': {'willingness to travel': valueArrayBrad[0,0,0],
                  'desire for new experience': valueArrayBrad[0,0,1],
                  'cost': valueArrayBrad[0,0,2],
                  #'indian food':
                  #'mexican food':
                  #'hipster points':
                  'vegitarian': valueArrayBrad[0,0,3]
                  }
          }          

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            

#here are some lists that show column keys and values
#print(peopleKeys)
#print(peopleValues)

M_people = np.array(peopleValues)
M_people.shape


# Next you collected data from an internet website. You got the following information.

#As above, I don't understand how we can support this proposed dictonary structure as it needs to have the same number of properties
#as people dictionary, so I will not use all properties.

#Generate random input for restaurant values.
valueArrayFlacos = np.random.randint(5, size=4)+1
valueArrayAnamias = np.random.randint(5, size=4)+1
valueArrayChilis = np.random.randint(5, size=4)+1
valueArrayBlueFish = np.random.randint(5, size=4)+1
valueArrayCheesecakeFactory = np.random.randint(5, size=4)+1
valueArrayTacoBell = np.random.randint(5, size=4)+1
valueArrayBurgerKing = np.random.randint(5, size=4)+1
valueArrayPanchos = np.random.randint(5, size=4)+1
valueArrayMcDonalds = np.random.randint(5, size=4)+1
valueArrayWingStop = np.random.randint(5, size=4)+1
restaurants  = {'Flacos':{'distance' : valueArrayFlacos[0],
                        'novelty' : valueArrayFlacos[1],
                        'cost':  valueArrayFlacos[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayFlacos[3]
                        },
              'Anamias':{'distance' : valueArrayAnamias[0],
                        'novelty' : valueArrayAnamias[1],
                        'cost':  valueArrayAnamias[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayAnamias[3]
                        },
              'Chilis':{'distance' : valueArrayChilis[0],
                        'novelty' : valueArrayChilis[1],
                        'cost':  valueArrayChilis[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayChilis[3]
                        },
              'BlueFish':{'distance' : valueArrayBlueFish[0],
                        'novelty' : valueArrayBlueFish[1],
                        'cost':  valueArrayBlueFish[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayBlueFish[3]
                        },
              'CheesecakeFactory':{'distance' : valueArrayCheesecakeFactory[0],
                        'novelty' : valueArrayCheesecakeFactory[1],
                        'cost':  valueArrayCheesecakeFactory[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayCheesecakeFactory[3]
                        },
              'TacoBell':{'distance' : valueArrayTacoBell[0],
                        'novelty' : valueArrayTacoBell[1],
                        'cost':  valueArrayTacoBell[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayTacoBell[3]
                        },
              'BurgerKing':{'distance' : valueArrayBurgerKing[0],
                        'novelty' : valueArrayBurgerKing[1],
                        'cost':  valueArrayBurgerKing[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayBurgerKing[3]
                        },
              'Panchos':{'distance' : valueArrayPanchos[0],
                        'novelty' : valueArrayPanchos[1],
                        'cost':  valueArrayPanchos[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayPanchos[3]
                        },
              'McDonalds':{'distance' : valueArrayMcDonalds[0],
                        'novelty' : valueArrayMcDonalds[1],
                        'cost':  valueArrayMcDonalds[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayMcDonalds[3]
                        },
              'WingStop':{'distance' : valueArrayWingStop[0],
                        'novelty' : valueArrayWingStop[1],
                        'cost':  valueArrayWingStop[2],
                        #'average rating': 
                        #'cuisine':
                        'vegitarians': valueArrayWingStop[3]
                        }
  
}


# Transform the restaurant data into a matrix(M_resturants) use the same column index.

restaurantsKeys, restaurantsValues = [], []

for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

#here are some lists that show column keys and values
#print(restaurantsKeys)
#print(restaurantsValues)

#len(restaurantsValues)
#reshape to 5 rows and 4 columns

M_restaurants = np.reshape(restaurantsValues, (10,4))
M_restaurants
M_restaurants.shape

# The most imporant idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.

#In a linear combination we are multiplying each term by a constant and summing the results.
#This relates to our restaurant matrix because multiplying our people matrix by the restaurant matrix
#will give us a "restaurant score" for each person that we can use to optimize lunch choice.

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent. 
#We will compute Janes scores for each restaurant.
flacoScore = M_people[0,0] * M_restaurants[0,0] + M_people[0,1] * M_restaurants[0,1] + M_people[0,2] * M_restaurants[0,2] + M_people[0,3] * M_restaurants[0,3]
anamiasScore = M_people[0,0] * M_restaurants[1,0] + M_people[0,1] * M_restaurants[1,1] + M_people[0,2] * M_restaurants[1,2] + M_people[0,3] * M_restaurants[1,3]
chilisScore = M_people[0,0] * M_restaurants[2,0] + M_people[0,1] * M_restaurants[2,1] + M_people[0,2] * M_restaurants[2,2] + M_people[0,3] * M_restaurants[2,3]
blueFishScore = M_people[0,0] * M_restaurants[3,0] + M_people[0,1] * M_restaurants[3,1] + M_people[0,2] * M_restaurants[3,2] + M_people[0,3] * M_restaurants[3,3]
cheesecakeFactoryScore = M_people[0,0] * M_restaurants[4,0] + M_people[0,1] * M_restaurants[4,1] + M_people[0,2] * M_restaurants[4,2] + M_people[0,3] * M_restaurants[4,3]
tacoBellScore = M_people[0,0] * M_restaurants[5,0] + M_people[0,1] * M_restaurants[5,1] + M_people[0,2] * M_restaurants[5,2] + M_people[0,3] * M_restaurants[5,3]
burgerKingScore = M_people[0,0] * M_restaurants[6,0] + M_people[0,1] * M_restaurants[6,1] + M_people[0,2] * M_restaurants[6,2] + M_people[0,3] * M_restaurants[6,3]
panchosScore = M_people[0,0] * M_restaurants[7,0] + M_people[0,1] * M_restaurants[7,1] + M_people[0,2] * M_restaurants[7,2] + M_people[0,3] * M_restaurants[7,3]
mcDonaldsScore = M_people[0,0] * M_restaurants[8,0] + M_people[0,1] * M_restaurants[8,1] + M_people[0,2] * M_restaurants[8,2] + M_people[0,3] * M_restaurants[8,3]
wingStopScore = M_people[0,0] * M_restaurants[9,0] + M_people[0,1] * M_restaurants[9,1] + M_people[0,2] * M_restaurants[9,2] + M_people[0,3] * M_restaurants[9,3]

janeScoreArray = np.array([flacoScore, anamiasScore, chilisScore, blueFishScore, cheesecakeFactoryScore, tacoBellScore, burgerKingScore, panchosScore, mcDonaldsScore, wingStopScore])
maxJaneScore = np.amax(janeScoreArray)
indexMaxJaneScore = np.where(janeScoreArray == maxJaneScore)
try:    
    print(f"Jane's max restaurant score is {maxJaneScore}, which is {list(restaurants.keys())[indexMaxJaneScore[0][0]]}.")
except:
    print("Crazy bizarro world where there were duplicate max scores for Jane. Re-run program.")

# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

M_new_people = np.swapaxes(M_people, 0, 1)
M_usr_x_rest = np.matmul(M_restaurants, M_new_people)
print(M_usr_x_rest)

print("In M_usr_x_rest, the rows represent restaurants and the columns represent people.")

# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

print(np.sum(M_usr_x_rest, axis=1))
print("The entrys represent the total score for each restaurant.")

maxRestaurantScore = np.amax(np.sum(M_usr_x_rest, axis=1))
indexMaxRestaurantScore = np.where(np.sum(M_usr_x_rest, axis=1) == maxRestaurantScore)
try:    
    print(f"Max restaurant score is {maxRestaurantScore}, which is {list(restaurants.keys())[indexMaxRestaurantScore[0][0]]}.")
except:
    print("Crazy bizarro world where there were duplicate max scores for restaurants. Re-run program.")


# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.  

sortedResults = M_usr_x_rest.argsort()[::-1]
sortedResults


np.sum(sortedResults, axis=1)


temp = M_usr_x_rest.argsort() 
M_usr_x_rest_rank = np.arange(len(M_usr_x_rest))[temp.argsort()]+1

#compare ranks to results                 
M_usr_x_rest                
print(np.sum(M_usr_x_rest, axis=1))            
print(np.sum(M_usr_x_rest_rank, axis=1))


# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
print("There is a difference because utimately the entire group of people have a ranking score that adds up to a constant value.",
      "This represents the real world situation where taking all aspects into account, the entire group cannot make an optimal choice", 
      " as each restaurant is optimal for a sub group of people.")

# How should you preprocess your data to remove this problem. 
print("We should pre-process our data so we take groups of people into account.")
# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.  

# Should you split in two groups today? 

# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
print("If the boss is paying, then cost consideration is no longer a factor. We should remove this property from our scoring.")
# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 




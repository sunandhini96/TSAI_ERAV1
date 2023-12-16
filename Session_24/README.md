# Task 1:

## We will implement value iteration and Q-learning. We will test our agents first on Gridworld.

## Pseudo code:
1. __init__  : Initializing the Q-values 
2. getQValue : 
 this function:

if it seen a state:

   return Q of state and action

else it didn't seen a state

   return 0 
   
3. computeValueFromQValue :
```
   -> actions: Retrieve the legal actions for the given state
   -> # If there are no legal actions (terminal state)
            # Return 0.0 as there are no actions to compute from
   -> Initialize the maximum value to negative infinity
   -> Iterate through each action in actions:
    Get the Q-value for the action in the state
    If the current Q-value is greater than or equal to the current max value
          Update the maximum value with the current Q-value

   Return the maximum Q-value among the legal actions
```

4. computeActionFromQValues:
```
    -> # Retrieve legal actions for the given state

    -> if no actions:  # If there are no legal actions (terminal state)
             # Return None as there are no actions to compute from

    -> Initialize the maximum value to negative infinity
     # Initialize the best action variable
    -> Iterate through each action in actions:
        # Get the Q-value for the action in the state

        # If the current Q-value is greater than or equal to the current max value
            # Update the maximum value with the current Q-value
            # Update the best action with the current action

   Return the action associated with the maximum Q-value
```
5. getAction:
```
    -> # Retrieve legal actions for the given state
    ->  # Initialize action variable to None

    -> # If there are no legal actions (terminal state)
               # Return None as there are no actions to compute from

    -> if  # With probability epsilon
                # Choose a random action
    -> else:  # Otherwise
                # Compute the best policy action

     Return the chosen action
```
6. update: we are updating Q value by using formula.
```
    updatedQValue = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
    self.values[(state, action)] = updatedQValue
```

## Output: <img width="1280" alt="era_s_24" src="https://github.com/sunandhini96/TSAI_ERAV1/assets/63030539/bd1cba62-8422-4b0a-b84a-5ae9fd118e51">
   
## Task 2:  

Our goal in this project is to use an image-based map of a city with roads and a car within this environment. 

There are two main objectives:

Keep the car on the roads as much as possible: Ensure the car stays on the designated roads within the city map. If the car moves onto other areas (not roads), it should be guided back to the road.

Reach the goal in the fewest steps: Guide the car to a specified destination within the city map using the roads available. The aim is to reach this destination as quickly as possible, minimizing the number of steps (or moves) taken by the car to reach the goal.

In simple terms, I want the car to stick to the roads and navigate efficiently to reach the desired destination within the city map. This involves keeping the car on the roads and steering it toward the goal in the fewest moves possible.

Youtube video link (https://youtu.be/3qZBVBmfxJU)


#  Contributors :

## Gosula Sunandini 
github repository : https://github.com/sunandhini96
## Katipally Vigneshwar Reddy
github repository : https://github.com/katipallyvig8899

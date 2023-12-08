# Assignemnet 1:
## Pseudo code:
1. __init__  : Initializing the Q-values 
2. getQValue : this function returns the Q of state and action and if it didn't seen a state return 0 .
3. computeValueFromQValue :
   -> actions: Retrieve the legal actions for the given state
   -> if no actions:  # If there are no legal actions (terminal state)
        return 0.0  # Return 0.0 as there are no actions to compute from
   -> Initialize the maximum value to negative infinity
   -> Iterate through each action in actions:
        qValue = getQValue(state, action)  # Get the Q-value for the action in the state
        if qValue >= maxValue or maxValue is negative infinity:
            # If the current Q-value is greater than or equal to the current max value
            maxValue = qValue  # Update the maximum value with the current Q-value

        return maxValue  # Return the maximum Q-value among the legal actions

4. computeActionFromQValues:

    -> actions = getLegalActions(state)  # Retrieve legal actions for the given state

    -> if no actions:  # If there are no legal actions (terminal state)
        return None  # Return None as there are no actions to compute from

    -> Initialize the maximum value to negative infinity
    bestAction = ""  # Initialize the best action variable
    -> Iterate through each action in actions:
    for each action in actions:
        qValue = getQValue(state, action)  # Get the Q-value for the action in the state

        if qValue >= maxValue or maxValue is negative infinity:
            # If the current Q-value is greater than or equal to the current max value
            maxValue = qValue  # Update the maximum value with the current Q-value
            bestAction = action  # Update the best action with the current action

    return bestAction  # Return the action associated with the maximum Q-value

5. getAction:

    -> legalActions = getLegalActions(state)  # Retrieve legal actions for the given state
    -> action = None  # Initialize action variable

    -> if no legalActions:  # If there are no legal actions (terminal state)
        return None  # Return None as there are no actions to compute from

    -> if flipCoin(self.epsilon):  # With probability epsilon
        action = random.choice(legalActions)  # Choose a random action
    -> else:  # Otherwise
        action = computeActionFromQValues(state)  # Compute the best policy action

        return action  # Return the chosen action

6. update: we are updating Q value by using formula.

    updatedQValue = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
    self.values[(state, action)] = updatedQValue
   

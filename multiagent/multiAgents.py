# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions, Grid  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        newPos = successorGameState.getPacmanPosition()
        
        newFood = successorGameState.getFood().asList() # this is from the Grid class
        
        foodDistances = [(manhattanDistance(newPos, food)) for food in newFood]

        minFoodDist = min(foodDistances) if foodDistances else 0
        
        nearbyGhosts = len([manhattanDistance(newPos, ghost) for ghost in successorGameState.getGhostPositions() if manhattanDistance(newPos, ghost) <=3])
        
        return successorGameState.getScore() - 4*nearbyGhosts +10.0/(minFoodDist + 1)
        
        
      
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    '''
    def terminate(self, gameState, depth):
    
    def getNextPlayer(self, currAgent, numAgents):
    
    
    '''
    def nextAg(self, currAg, numAgs):
        '''return the next agent and whether to increase depth'''
        if currAg != numAgs:
            return currAg + 1,False
        else:
            return 0,True    
    
    def minimax(self, gameState, currPlayer, depth):
        val_action = {'val': None, 'action': None} #store current val and action in a dict.
        
        #val_action['val'] = -float('inf') if currPlayer == 0 else float('inf')
        
        legalActions = gameState.getLegalActions(currPlayer)
        numAgents = gameState.getNumAgents() -1
        nextPlayer,nextDepth = self.nextAg(currPlayer, numAgents)
       
        #first check terminal states or when depth is reached
        #in this case, there's no action to be made, so return action = None
        if gameState.isWin() or gameState.isLose() or depth == 0: 
            val_action['val'] = self.evaluationFunction(gameState)
            return val_action
        
        if nextDepth: #this means it's pacman's turn again 
            depth -= 1
          
        
        
        if currPlayer == 0:
            
            vals = [self.minimax(gameState.generateSuccessor(currPlayer, action), nextPlayer,depth)['val'] for action in legalActions]
            
            maxval = max(vals)
            
            index = vals.index(maxval)
            
            action = legalActions[index]
            
            val_action['val'], val_action['action'] = maxval, action
        
        else: 
            
            vals = [self.minimax(gameState.generateSuccessor(currPlayer, action), nextPlayer,depth)['val'] for action in legalActions]
            
            minval = min(vals)
            
            index = vals.index(minval)
            
            action = legalActions[index]
            
            val_action['val'], val_action['action'] = minval, action
            
        
        return val_action
        
            
            
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)['action'] 
       

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def nextAg(self, currAg, numAgs):
        '''return the next agent and whether to increase depth'''
        if currAg != numAgs:
            return currAg + 1,False
        else:
            return 0,True
        
    
    def alpha_beta(self, gameState, alpha, beta, depth, currPlayer):
        score_action = {'score': None, 'action': None} #store current score/val and action in a dict.
        
        score_action['score'] = float('-inf') if currPlayer == 0 else float('inf')
        
        legalActions = gameState.getLegalActions(currPlayer)
        numAgents = gameState.getNumAgents() -1
        nextPlayer,nextDepth = self.nextAg(currPlayer, numAgents)
        
        if gameState.isWin() or gameState.isLose() or depth == 0: 
            score_action['score'] = self.evaluationFunction(gameState)
            return score_action
        
        if nextDepth:
            depth -= 1
            
       
        for action in legalActions:  
            
            #check if we can prune
            if (score_action['score'] >= beta and currPlayer == 0) or (score_action['score'] <= alpha and currPlayer != 0):
                return score_action 
            
            newVal = self.alpha_beta(gameState.generateSuccessor(currPlayer,action),alpha, beta, depth, nextPlayer)['score']
            
                
            if (currPlayer == 0 and newVal > score_action['score']) or (currPlayer != 0 and newVal < score_action['score']):
                score_action['score'] = newVal
                score_action['action'] = action
                
                alpha = max(score_action['score'], alpha) if currPlayer == 0 else alpha 
                beta = min(score_action['score'], beta) if currPlayer != 0 else beta
                  
                
        return score_action
                
                
                
                
            
                
    
    def getAction(self, gameState):
        return self.alpha_beta(gameState, float('-inf'), float('inf'), self.depth,0)['action']
    
    
        
        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def nextAg(self, currAg, numAgs):
        '''return the next agent and whether to increase depth'''
        if currAg != numAgs:
            return currAg + 1,False
        else:
            return 0,True    
    
    
    def expectiMax(self, gameState,depth,currPlayer):
        
        score_action = {'score': None, 'action': None} #store current score (val) and action (best move) in a dict.
                
        #score_action['score'] = -float('inf') if currPlayer == 0 else 0     
                
        legalActions = gameState.getLegalActions(currPlayer)
        numAgents = gameState.getNumAgents() -1
        nextPlayer,nextDepth = self.nextAg(currPlayer, numAgents)
        
        if gameState.isWin() or gameState.isLose() or depth == 0: 
            score_action['score'] = self.evaluationFunction(gameState)
            return score_action
        
        if nextDepth:
            depth -= 1
            
        prob = (len(legalActions))**-1
        
        if currPlayer == 0:
            vals = [self.expectiMax(gameState.generateSuccessor(currPlayer, action), depth,nextPlayer)['score'] 
                    for action in legalActions]
            maxVal = max(vals)
            
            index = vals.index(maxVal) 
            
            action = legalActions[index]
            
            score_action['score'], score_action['action'] = maxVal, action
        else:
            vals = [self.expectiMax(gameState.generateSuccessor(currPlayer, action), depth,nextPlayer)['score'] 
                    for action in legalActions]
            
            score_action['score'] = sum(vals)*prob
        
        return score_action
    
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectiMax(gameState,self.depth,0)['action']
        
        
   
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      
      Description: My idea was to just use two pieces of info: the minimum
      distance from the pacman to a piece of food (if one exists) and then 
      how many ghosts are "nearby" the pacman (which I somewhat arbitrarily chose
      to be less than 3 manhattan distance, because it seems to work well).
      The min food distance is important because a rational pacman should 
      go and eat the food closest to it, so it wants to make min food dist 0 --
      which is why I used the reciprocal of this value multiplied by a constant.
      And the penalty for having nearby ghosts is self-explanatory: we want the 
      pacman to move away from nearby ghosts, but essentially it can disregard
      the ghosts and just focus on eating food if there are no ghosts nearby. 
      
      note: I just used my original eval function bc it seems to hold up here
      pretty well too
    """
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    
    foodDistances = [(manhattanDistance(newPos, food)) for food in newFood]
    
    minFoodDist = min(foodDistances) if foodDistances else 0
    
    nearbyGhosts = len([manhattanDistance(newPos, ghost) for ghost in currentGameState.getGhostPositions() if manhattanDistance(newPos, ghost) <=3])
        
 
    return currentGameState.getScore() - 4*nearbyGhosts +10.0/(minFoodDist + 1)
            
    

# Abbreviation
better = betterEvaluationFunction

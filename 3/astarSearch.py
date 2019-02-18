class Node:
    def __init__(self, state, f=0, g=0 ,h=0):
        self.state = state
        self.f = f
        self.g = g
        self.h = h
    def __repr__(self):
        return "Node(" + repr(self.state) + ", f=" + repr(self.f) + \
               ", g=" + repr(self.g) + ", h=" + repr(self.h) + ")"

def aStarSearch(startState, actionsF, takeActionF, goalTestF, hF):
    h = hF(startState)
    print(h)
    startNode = Node(state=startState, f=0+h, g=0, h=h)
    return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'))

def aStarSearchHelper(parentNode, actionsF, takeActionF, goalTestF, hF, fmax):
    if goalTestF(parentNode.state):
        return ([parentNode.state], parentNode.g)
    ## Construct list of children nodes with f, g, and h values
    actions = actionsF(parentNode.state)

    print(actions)
    if not actions:
        return ("failure", float('inf'))
    children = []
    for action in actions:
        (childState,stepCost) = takeActionF(parentNode.state, action)
        print(childState)
        print(stepCost)
        h = hF(childState)
        print(h)
        g = parentNode.g + stepCost
        print(g)
        f = max(h+g, parentNode.f)
        print(f)
        childNode = Node(state=childState, f=f, g=g, h=h)
        print(childNode)
   
        children.append(childNode)
    while True:
        # find best child
        children.sort(key = lambda n: n.f) # sort by f value
        bestChild = children[0]
        if bestChild.f > fmax:
            return ("failure",bestChild.f)
        # next lowest f value
        alternativef = children[1].f if len(children) > 1 else float('inf')
        # expand best child, reassign its f value to be returned value
        result,bestChild.f = aStarSearchHelper(bestChild, actionsF, takeActionF, goalTestF,
                                            hF, min(fmax,alternativef))
        if result is not "failure":               #        g
            result.insert(0,parentNode.state)     #       / 
            return (result, bestChild.f)          #      d
                                                  #     / \ 
if __name__ == "__main__":                        #    b   h   
                                                  #   / \   
    successors = {'a': ['b','c'],                 #  a   e  
                  'b': ['d','e'],                 #   \         
                  'c': ['f'],                     #    c   i
                  'd': ['g', 'h'],                #     \ / 
                  'f': ['i','j']}                 #      f  
    def actionsF(s):                              #       \ 
        try:                                      #        j
            ## step cost of each action is 1
            return [(succ,1) for succ in successors[s]]
        except KeyError:
            return []
    def takeActionF(s,a):
        return a
    def goalTestF(s):
        return s == goal
    def h1(s):
        return 0
    start = 'a'
    goal = 'h'
    result = aStarSearch(start,actionsF,takeActionF,goalTestF,h1)
    print('Path from a to h is', result[0], 'for a cost of', result[1])

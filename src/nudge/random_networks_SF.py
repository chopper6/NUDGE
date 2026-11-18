from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import pickle

totalLogicDic = pickle.load(open("../input/networks/cellCollective_logical_functions.p", 'rb'))

#%%
# For pareto distribution
a = 100. # shape
m = 1.  # mode
def main(nodeNum, minIndegree, maxIndegree, random_seed=False):
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    allNodeLength = nodeNum
    data = (np.random.pareto(a, allNodeLength) + 1) * m
    data = np.reshape(data, (-1, 1))
    scaler = MinMaxScaler(copy=True, feature_range=(minIndegree, maxIndegree))  # Column을 기준으로 한다.
    scaler.fit(data)
    data = scaler.transform(data)
    data = np.round(data)
    data = np.reshape(data, (-1))
    data = data.tolist()
    indegreeList = list(map(int, data))
    indegreeTotalLength = sum(indegreeList)

    fillNumber = len(str(allNodeLength))
    allNodes = ["x" + str(i).zfill(fillNumber) for i in range(1, allNodeLength + 1)] #node form
    formatAttr = ""
    formatNormal = ""

    for k in allNodes:
        initialStateLine = k + " = Random" + "\n"
        formatAttr = formatAttr + initialStateLine
    formatAttr = formatAttr + "\n\n"

    for node, indegree in zip(allNodes, indegreeList):

        selectedNodes = random.sample(allNodes, indegree)
        biologicalRandomLogic = random.choice(totalLogicDic[str(indegree)])

        for n, selectNode in enumerate(selectedNodes):
            existingNode = "z" + str(n + 1).zfill(2)
            biologicalRandomLogic = biologicalRandomLogic.replace(existingNode, selectNode)

        biologicalRandomLogic = biologicalRandomLogic.replace("~", "! ")
        formatNormal = formatNormal + node + ", " + biologicalRandomLogic + "\n"
        formatAttr = formatAttr + node + " *= " + biologicalRandomLogic + "\n"

    return formatAttr, formatNormal, indegreeTotalLength

#formatAttr, formatNormal, indegreeTotalLength = main(12, 1, 4)
#print(formatNormal)

#formatNormal
for i in range(0,10):    
    formatAttr, formatNormal, indegreeTotalLength = main(3, 1, 3)   #(node개수, 최소 degree 개수, 최대 degree 개수)
    f = open('RBN(3,1,3)%d.txt' % (i+1), 'w')
    #f.write('targets, factors\n')
    f.write(formatNormal)
    f.write('\n')
    f.close()
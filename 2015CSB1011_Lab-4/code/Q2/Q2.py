import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import skimage.io as skio

from collections import defaultdict

import math
import heapq
import random
 
import networkx as nx


def ProcessPQ(joints, marg, feature_length):
  """
  Populates a heap in descending order of mutual informations.
  This is used to build the maximum spanning tree.
  Contains mutual information of every feature with respect to every other feature
  """
  #variable defining the heap
  pq = []

  for i in range(feature_length):
	for j in range(i+1, feature_length):
	  I = 0
	  for x_u, p_x_u in marg[i].iteritems():
		for x_v, p_x_v in marg[j].iteritems():
		  if (x_u, x_v) in joints[(i, j)]:
			p_x_uv = joints[(i, j)][(x_u, x_v)]
			I += p_x_uv * (math.log(p_x_uv, 2) - math.log(p_x_u, 2) - math.log(p_x_v, 2))
	  heapq.heappush(pq, (-I, i, j))
  
  return pq


def findSet(parent, i):
  
  while i != parent[i]:
	i = parent[i]

  return i

def buildMST(pq, feature_length):
  """
  Builds the MST using the heap generated above.
  It returns the edges that needs to be connected using the highest mutual information
  """
  
  parent = range(feature_length)
  size = [1]*feature_length

  count = 0
  edges = set()
  while count < feature_length-1:
	item = heapq.heappop(pq)
	i = item[1]
	j = item[2]
	seti = findSet(parent, i)
	setj = findSet(parent, j)
	if seti != setj:
	  if size[seti] < size[setj]:
		size[setj] += size[seti]
		parent[seti] = setj
	  else:
		size[seti] += size[setj]
		parent[setj] = seti
	  edges.add((i, j))
	  count += 1

  return edges

G2 = None
pos2 = None

def buildVisual(edges, feature_length, labels, fname, title=None):
  
  """
  Tree built could be visualized. 
  This is just for visual perspectives.
  Saves the graph
  """

  global G2
  global pos2

  if type(G2) == type(None):
	G = nx.Graph()
	for i in range(feature_length):
	  G.add_node(i)
	pos = nx.spring_layout(G, k=10., scale = 10)
	G2 = G
	pos2 = pos
  else:
	G = G2
	pos = pos2

  nx.draw_networkx_nodes(G, pos, node_size=1000)

  nx.draw_networkx_labels(G, pos,labels,font_size=8)
  nx.draw_networkx_edges(G, pos, edgelist=list(edges))
  if title:
	plt.title(title)
  plt.savefig(fname)
  plt.close()


# Function to perform Gibbs Sampling
def Gibbs_Sampler(N_Samples,N_Iter, feature_length, marg, joints, triplets, edges):

  Samples = [] # Initializing array of samples as empty

  for i in range(0, N_Samples):
	
	# Generating a random assignment
	Assignment = []

	for j in range(0,feature_length):
	  # Choosing a random value for each attribute
	  val = random.choice(marg[j].keys())
	  Assignment.append(val)

	# N_Iter = 1000  # Generate each sample after 1000 iterations
  
	for j in range(0, N_Iter):
	  
	  # Selecting a variable at random to sample
	  var = random.randint(0, feature_length-1)

	  # Finding parents of the variable from the Bayesian Network
	  Parents = []
	  Variables = []
	  for E in edges:
		if(E[1] == var):
		  Parents.append(E[0])

	  if(len(Parents) > 0):
		Parents.sort()
		Variables = list(Parents)
	  
	  Variables.append(var)
	  Variables.sort()

	  # Generating Sample Probability
	  sample_prob = random.uniform(0,1)
	  prob = 0

	  # Finding an appropriate assignment for variable according to sample probabilty
	  for val in marg[var].keys():
		
		# Fixing the values of parents according to assignment
		Variable_Val = []
		Parent_Val = []

		for variable in Variables:
		  if(variable == var):
			Variable_Val.append(val)
		  else:
			Variable_Val.append(Assignment[variable])
			Parent_Val.append(Assignment[variable])

		# Note that number of parents <= 2

		# Case 1 : Number of parents = 0
		if(len(Parents) == 0):
		  prob += marg[var][val]

		# Case 2 : Number of parents = 1
		elif(len(Parents) == 1):
		  # if P(Parents(Xi)) != 0
		  if(marg[Parents[0]][Parent_Val[0]] != 0):
			prob += joints[tuple(Variables)][tuple(Variable_Val)]/marg[Parents[0]][Parent_Val[0]]

		# Case 3 : Number of parents = 2
		elif(len(Parents) == 2):
		  # if P(Parents(Xi)) != 0
		  if(joints[tuple(Parents)][tuple(Parent_Val)] != 0):
			prob += triplets[tuple(Variables)][tuple(Variable_Val)]/joints[tuple(Parents)][tuple(Parent_Val)]

		# Checking if probabilty of sample > sample probability threshold
		if(prob > sample_prob):
		  Assignment[var] = val
		  break

	# Appending to the list of samples generated
	Samples.append(Assignment)
	
  return Samples


#Returns the probability estimate for each test set instance from Gibbs's sample set
def Likelihood_Gibbs(test_samples, size_test, Samples, N_Samples):

  Probability = []

  for TS in test_samples:
	count = 0

	# Finding count of test sample in Gibbs Samples
	for GS in Samples:
	  if(TS == GS):
		count += 1

	prob = float(count)/N_Samples

	Probability.append(prob)

  return Probability


# Returns the ground truth probabilities for test instances
def Estimate_GroundTruth(test_samples):

	test_samples_unique = []
  
	GroundTruth = []

	size_test =len(test_samples)

	for i in range(0,size_test):
		
  		if test_samples[i] in test_samples_unique:	
			continue

  		else:
  			test_samples_unique.append(test_samples[i])
  			
  			count = 1
  			
  			for j in range(i+1,size_test):
  				if test_samples[i]==test_samples[j]:
  					count+=1	

  			GroundTruth.append(float(count)/size_test)		


  	return test_samples_unique,GroundTruth	

##############main#####################

labels = {0: "Age",
		  1: "Workclass",
		  2: "education",
		  3: "education-num",
		  4: "marital-status",
		  5: "occupation",
		  6: "relationships",
		  7: "race",
		  8: "sex",
		  9: "capital-gain",
		  10: "capital-loss",
		  11: "hours-per-week",
		  12: "native-country",
		  13: "salary",
		 }

f = open("data.txt", "r")
triplets = {}
joints = {}
marg = {}

feature_length = 14
data_size = 25000

for i in range(feature_length):
  marg[i] = defaultdict(float)

  for j in range(i+1, feature_length):
	joints[(i, j)] = defaultdict(float)

	for k in range(j+1, feature_length):
	  triplets[(i, j, k)] = defaultdict(float) 

count_aggr = 0

#Reading of file
for line in f:
  n = line.strip().split("  ")
  
  count_aggr += 1

  #Calculates the marginal and joint distributions of the dataset
  #Each marginal feature has a dictionary telling the probability of getting that value  
  #For each of the pair of features what is the probability of getting the two value pair together
  for i in range(feature_length):
	marg[i][n[i]] += 1./data_size

	for j in range(i+1, feature_length):
	  joints[(i,j)][(n[i], n[j])] += 1./data_size

	  for k in range(j+1, feature_length):
		triplets[(i, j, k)][(n[i], n[j], n[k])] += 1./data_size 

#Reading end

pq = ProcessPQ(joints, marg, feature_length)
edges = buildMST(pq, feature_length)
# buildVisual(edges, feature_length, labels, "final.jpg", title="%d samples"%data_size)

print (edges),'\n'

#Consider p is parent of q
#You can use the joint probabilities and marginal probabilities calculated above in your code 
#Write your code here

N_Samples = 25000 # Number of samples

N_Iter = 400# Number of iterations to generate one gibbs sample

Samples = Gibbs_Sampler(N_Samples,N_Iter, feature_length, marg, joints, triplets, edges)

# Storing Samples in a text file
fout = open("samples", "w")

for i in range(0, N_Samples):
  string = ""
  for temp in Samples[i]:
	string += str(temp) + " " 
  
  fout.write(string + "\n")

fout.close()



# Opening test set file  
f_test = open("data_test.txt", "r")

# Reading test set file
test_samples = []

size_test = 0

for line in f_test:
  n = line.strip().split("  ")
  test_samples.append(n)
  size_test += 1

# Call to get Gibbs sample set
test_samples_unique,GroundTruth = Estimate_GroundTruth(test_samples)


GibsProb = Likelihood_Gibbs(test_samples_unique, size_test, Samples, N_Samples)


file = open("probs.txt","w")


#Calculating error in probabilities
error=0

for i in range(0,len(GroundTruth)):

	file.write(str(GroundTruth[i])+" "+str(GibsProb[i])+"\n");

	error += abs(GroundTruth[i] - GibsProb[i])

print 'The error in probabilities is ',error

file.close()
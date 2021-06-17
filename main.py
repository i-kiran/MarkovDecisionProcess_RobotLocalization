import numpy as np 
import random
#initializing probability statematrix from given transisition probabilities
#|s1-s1   s2-s1   s3-s1   s4-s1   s5-s1   s6-s1|
#|s1-s2   s2-s2   s3-s2   s4-s2   s5-s2   s6-s2|
#|s1-s3   s2-s3   s3-s3   s4-s3   s5-s3   s6-s3|
#|s1-s4   s2-s4   s3-s4   s4-s4   s5-s4   s6-s4|
#|s1-s5   s2-s5   s3-s5   s4-s5   s5-s5   s6-s5|
#|s1-s6   s2-s6   s3-s6   s4-s6   s5-s6   s6-s6|
States = np.array([[0.2,0.4,0,0,0,0],[0.8,0.2,0.27,0,0,0],[0,0.4,0.2,0.4,0,0.27],[0,0,0.27,0.2,0.8,0],[0,0,0,0.27,0.2,0],[0,0,0.27,0,0,0.2]]) 
#Assuming uniform probability distribution for start state given 6 tiles/states
b = 1/6
Start = np.array([b,b,b,b,b,b])
e = 0.25 #given in the error rate
#created a sensor matrix that evaluates the probability for evidence 
def Sensor(States , Start , e , n): 
	s = []
	for i in n:
		x = ((1-e)**(4 - i)) 
		y = (e**i)
		p = x*y
		s.append(p)
		size = len(s)
		evidence_matrix = np.zeros((size , size))
		np.fill_diagonal(evidence_matrix , s)
	return evidence_matrix
#used filtering to estimate state in time t given state in time t-1 and evidence in t
def filter(States , Start , evidence_matrix):
	newstate=np.dot(evidence_matrix , np.dot(States , Start))
	s = np.sum(newstate)
	n_newstate = newstate/s
	return n_newstate

#four possible obstacle readings of robot are :
#defining moving constraints vector for each tile
a = [4,1,0,1,4,4] 
b = [3,2,1,0,3,3]
c = [3,0,1,2,3,3]
d = [0,3,4,3,0,0]
up = Sensor(States , Start , e , a) #for tile3
up_right = Sensor(States , Start , e , b)#for tile 4
up_left = Sensor(States , Start , e ,c)#for tile 2
down_right_left = Sensor(States , Start , e ,d) #for tile1,5,6
li = [down_right_left , up_right , up , up_left , down_right_left , down_right_left , up_right , up , up_left , up_right]

for i in range(10):
	random.shuffle(li)
	for l in li:
		pos10 = filter(States , Start , l)
	pos = pos10
	#print(pos)
	m = pos.argmax()
	print('chosen tile : ',m+1,'with probability : ',pos[m])

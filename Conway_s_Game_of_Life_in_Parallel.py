''' Conway's Game of Life Rules:
1. Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
2. Any live cell with two or three live neighbours lives on to the next generation.
3. Any live cell with more than three live neighbours dies, as if by overpopulation.
4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
'''

from mpi4py import MPI
from PIL import Image
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
rowCount = 0
columnCount = 0

def getNeighbors(x, y, xmax, ymax):
	x2, y2 = x, y
	neighbors = [] # contains coordinates of all neighbors of x,y

	for x2 in range(x-1, x+2):
		for y2 in range(y-1, y+2):
			if (-1 < x <= xmax-1 and
				-1 < y <= ymax-1 and
				(x != x2 or y != y2) and
				(0 <= x2 <= xmax-1) and
				(0 <= y2 <= ymax-1)):
					neighbors.extend([[x2, y2]])

	return neighbors

def getNextGraph(graph):
	rowCount = len(graph) # number of rows received
	columnCount = len(graph[0])	# number of columns received
	nextRows = graph.copy() # performs a deep copy in this case

	for row in range(rowCount):
		for column in range(columnCount):
			liveNeighbors = 0
			neighbors = getNeighbors(row, column, rowCount, columnCount) # get the coordinates of every neighbor
			
			for coordinates in neighbors:
				if (graph[coordinates[0]][coordinates[1]] == 1):
					liveNeighbors += 1

			if (liveNeighbors < 2 or liveNeighbors > 3):
				nextRows[row][column] = 0 # Rule 1 and 3
			elif (liveNeighbors == 3):
				nextRows[row][column] = 1 # Rule 4

	return nextRows

def playGame(graph):
	# first rank zero distributes instructions to other ranks:
	if (rank == 0):
		rowCount = len(graph) # number of rows in file
		columnCount = len(graph[0])	# number of columns in file (assumes graph not jagged)

		# divvy up partitions of graph by rows:
		division = int(rowCount / (size))
		start = 0
		end = division
		partitions = []

		# each rank will get an extra row above and below its partition if present:
		for i in range(0, size):
			if (size == 1): # only 1 rank
				partitions.append(graph[start:end])
			elif (i == 0 and size > 1):
				partitions.append(graph[start:end+1])
			elif (i == 1 and size == 2):
				partitions.append(graph[start-1:end])
			elif (size > 2):
				if (i != size-1):
					partitions.append(graph[start-1:end+1])
				else: # last rank does not get additional row on the end
					partitions.append(graph[start-1:end])
			start += division
			end += division
	else:
		partitions = None

	# assign ranks partitions of the graph through scattering:
	partitions = comm.scatter(partitions, root=0)

	# now instruct the ranks with the scattered information:
	partitions = getNextGraph(partitions)

	# get rid of any extra rows that the partitions have:
	if (rank == 0 and size > 1):
		partitions = numpy.delete(partitions, len(partitions)-1, 0)
	elif (rank == 1 and size == 2):
		partitions = numpy.delete(partitions, 0, 0)
	elif (rank != size-1 and size > 2):
		partitions = numpy.delete(partitions, 0, 0)
		partitions = numpy.delete(partitions, len(partitions)-1, 0)
	elif (rank == size-1 and size > 2):
		partitions = numpy.delete(partitions, 0, 0)


	#print ('Rank {} of {} on {} has {}'.format(rank, size-1, name, partitions))

	# gather new rows containing dead and alive cells from processes:
	nextGraph = comm.gather(partitions, root=0)
	newGraph = [] # for use of going to the next step in total game

	if (rank == 0):
		# print the next graph: 
		for rows in nextGraph:
			for row in rows:
				newGraph.append(row)
				for column in row:
					#print(column, ' ', end='')
					pass
				#print()
		#print('\n')
	
	newGraph = numpy.array(newGraph) # convert 2D array to numpy array
	#creating a pixel image of the game board. White means alive
	img = Image.frombytes('1', (rowCount,columnCount), newGraph, decoder_name='raw')
	img.show()
	return newGraph

def main():
	NUMBER_OF_STEPS = 5
	inputFile = open("input.txt", "r")
	graph = []

	for row in inputFile: # creates 2D array of integers
		graph.append(list(map(int, row.rstrip().replace(' ',''))))

	inputFile.close()

	graph = numpy.array(graph) # convert 2D array to 2D numpy arrays
	nextGraph = playGame(graph)

	for i in range(NUMBER_OF_STEPS-1):
		nextGraph = playGame(nextGraph)

if __name__ == '__main__':
	main()
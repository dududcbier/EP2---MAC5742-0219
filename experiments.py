from subprocess import getoutput 
import subprocess
import re
import pprint as pp
import numpy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def unit_to_expo_notation(x):
	return {
		'm': "E-3",
		'u': "E-6",
		'n': "E-9"
	}.get(x, x)

def seconds_to_float(s):
	if not s.endswith("s"): return None
	s = s.replace("s", "")
	return float(s.replace(s[-1], unit_to_expo_notation(s[-1])))

def get_total_time(times):
	total = 0
	for t in times.keys():
		total += times[t].time
	return total

class Line_Result:

	def __init__(self, s):
		l = line.split()
		if len(l) < 7: print("SOMETHING'S WRONG")
		self.time_percent = float(l[0].strip("%"))
		self.time = seconds_to_float(l[1])
		self.avg = seconds_to_float(l[3])
		self.min = seconds_to_float(l[4])
		self.max = seconds_to_float(l[5])
		self.name = " ".join(l[6:])
		self.count = 1

	def __str__(self):
		return " ".join([self.name, str(self.time_percent) + "%", str(self.time), str(self.avg), str(self.min), str(self.max)])

	def __repr__(self):
		return " ".join([self.name, str(self.time_percent) + "%", str(self.time), str(self.avg), str(self.min), str(self.max)])
		
	# def add(self, result):
	# 	if (self.name == result.name):
	# 		self.time_percent = (self.time_percent * self.count + result.time_percent * result.count) / (result.count + self.count)
	# 		self.time = (self.time * self.count + result.time) / (self.count + 1)
	# 		self.avg = (self.avg * self.count + result.avg) / (self.count + 1)
	# 		self.min = (self.min * self.count + result.min) / (self.count + 1)
	# 		self.max = (self.max * self.count + result.max) / (self.count + 1)
	# 		self.count += result.count
	# 	else:
	# 		print("Trying to add different types of results!")

p = re.compile('^[ ]*\d+\.?\d+?')
algorithms = ['rot_13', 'base64']
values = [True, False]
test_files = ['tale_of_two_cities.txt', 'moby_dick.txt',  'ulysses.txt', 'king_james_bible.txt',
 'hubble_2.png', 'mercury.png', 'hubble_3.tif', 'hubble_1.tif']

sequential_results = {}
cuda_results = {}

for a in algorithms:
	cuda_results[a] = {}
	sequential_results[a] = {}
	for f in test_files:
		for sequential in values:
			total_times = []
			times = {}
			for i in range(0, 10):
				if not sequential:
					output = getoutput("nvprof ./encode_cuda sample_files/" + f + " " + a).split("\n")
					for line in output:
						if (p.match(line)):
							t = Line_Result(line)
							times[t.name] = t
					total_times.append(get_total_time(times))
				else:
					start = time.time()
					subprocess.call(["./encode_cuda","sample_files/" + f, a, "-s"])
					total_times.append(time.time() - start)

			if (sequential):
				sequential_results[a][f] = total_times
			else:
				cuda_results[a][f] = total_times
			print("\nAlgorithm: {} seq = {}, file: {}".format(a, sequential, f))
			print("Average: {} seconds".format(numpy.mean(total_times)))
			print("Standard deviation: {}\n".format(numpy.std(total_times)))

# File Impact
for seq in values:
	for a in algorithms:
		plt.figure()
		data_to_plot = []
		for f in test_files:
			if ((a != 'rot-13') or (a == 'rot-13' and not f in ['mercury.png', 'hubble_1.tif', 'hubble_2.png', 'hubble_3.tif'])):
				if (seq): data_to_plot.append(sequential_results[a][f])
				else: data_to_plot.append(cuda_results[a][f])
		plt.boxplot(data_to_plot, meanline=True)
		plt.xlabel("Arquivo")
		plt.ylabel("Tempo de execução (s)")
		plt.xticks(list(range(1, len(test_files) + 1)), test_files, rotation=45)
		if not seq:
			plt.title("Tempo de execução usando " + a + " para diferentes arquivos")
		else:
			plt.title("Tempo de execução usando " + a + "_seq" + " para diferentes arquivos")
		name = a
		if (seq): name += "_seq"
		name += ".pdf"
		plt.tight_layout()
		try:
			plt.savefig('../graphs/file_impact/' + name)
		except FileNotFoundError:
			os.makedirs('../graphs/file_impact/')
			plt.savefig('../graphs/file_impact/' + name)

# CUDA impact
for f in test_files:
	for a in algorithms:
		data_to_plot = []
		for seq in values:
			if (seq): data_to_plot.append(sequential_results[a][f])
			else: data_to_plot.append(cuda_results[a][f])
			
		plt.figure()
		plt.boxplot(data_to_plot, meanline=True)
		plt.xlabel("Implementação")
		plt.ylabel("Tempo de execução (s)")
		plt.xticks([1, 2], ["Sequencial", "CUDA"])
		plt.title("Comparação do tempo de execução entre a versão em CUDA e sequencial de")
		name = f + ".pdf"
		plt.tight_layout()
		try:
			plt.savefig('../graphs/cuda_impact/' + a + "/" + name)
		except FileNotFoundError:
			os.makedirs('../graphs/cuda_impact/' + a)
			plt.savefig('../graphs/cuda_impact/' + a + "/" + name)
		plt.close()

















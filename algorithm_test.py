from subprocess import call
import filecmp

class bcolors:
	OK = '\033[92m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'

class Test_Error:

	def __init__(self, text, match, mismatch, errors, algorithm):
		self.text = text;
		self.match = match
		self.mismatch = mismatch;
		self.errors = errors;
		self.algorithm = algorithm

	def print(self):
		print(self.algorithm + ": ", end="")
		print(bcolors.FAIL + "Failed: "  + bcolors.ENDC + self.text)
		print("\tmatch: {}".format(self.match))
		print("\tmismatch: {}".format(self.mismatch))
		print("\terrors: {}".format(self.mismatch))

call(["make"])
algorithms = ['rot_13', 'base64', 'arcfour']
test_files = ['king_james_bible.txt', 'moby_dick.txt', 'tale_of_two_cities.txt', 'ulysses.txt']

# Note that the image files were not included in the tests. 
# Not even the sequential algorithms were able to write
# valid images files...

failed_tests = {}

print("Running tests...\n")
for a in algorithms:
	for f in test_files:
		call(["./encode_cuda", "sample_files/" + f, a, "-d", "-x"])
		call(["./encode_cuda", "sample_files/" + f, a, "-d", "-x", "-s"])

	match, mismatch, errors = filecmp.cmpfiles("encoded_files/sequential/" + a + "/", "encoded_files/" + a + "/", test_files);
	if (len(match) != len(test_files)):
		failed_tests[a] = Test_Error("Sequential algorithm should yield the same encoding for a file as the Cuda version", match, mismatch, errors, a)

	match, mismatch, errors = filecmp.cmpfiles("sample_files/", "decoded_files/" + a + "/", test_files);
	if (len(match) != len(test_files)):
		failed_tests[a] = Test_Error("Decoded file from encoded file should match the original file", match, mismatch, errors, a)

	if a not in failed_tests:
		print(a + ": " + bcolors.OK + "OK" + bcolors.ENDC)
	else:
		failed_tests[a].print()

print("")


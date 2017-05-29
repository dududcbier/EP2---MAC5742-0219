from subprocess import call
import filecmp

class bcolors:
	OK = '\033[92m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'

call(["make"])
algorithms = ['rot_13', 'base64']
test_files = ['ulysses.txt', 'king_james_bible.txt', 'moby_dick.txt', 'tale_of_two_cities.txt']
print("Running tests...\n")
for a in algorithms:
	for f in test_files:
		call(["./encode_cuda", "sample_files/" + f, a, "-d"])
	match, mismatch, errors = filecmp.cmpfiles("sample_files/", "decoded_files/" + a + "/", test_files);
	print(a + ": ", end="")
	if (len(match) == len(test_files)):
		print(bcolors.OK + "OK" + bcolors.ENDC)
	else:
		print(bcolors.FAIL + "Failed"  + bcolors.ENDC)
		print("\tmatch: {}".format(match))
		print("\tmismatch: {}".format(mismatch))
		print("\terrors: {}".format(mismatch))

print("")

from subprocess import call
import filecmp

call(["make"])
algorithms = ['rot_13']
test_files = ['ulysses.txt', 'king_james_bible.txt', 'moby_dick.txt', 'tale_of_two_cities.txt']

for a in algorithms:
	for f in test_files:
		call(["./encrypt_cuda", "sample_files/" + f, a, "-d"])

match, mismatch, errors = filecmp.cmpfiles("encrypted_files/", "decrypted_files/", test_files);

if (len(match) == len(test_files)):
	print("Passed!")
else:
	print("Test failed")
	print("mismatch: {}".format(mismatch))
	print("errors: {}".format(mismatch))

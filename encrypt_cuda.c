#include "rot-13.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <libgen.h>

#define ROT_13 0
#define ENCRYPTED_FILES_DIR "encrypted_files/"
#define DECRYPTED_FILES_DIR "decrypted_files/"

int main(int argc, char *argv[]) {
	FILE *f;
	char *filename;
	char enc_path[1000];
	long size;
	int algorithm = -1;
	struct stat st = {0};
	if (argc <= 2) {
		printf("usage: %s [filename] [algorithm] -d\n", argv[0]);
		printf("\tfilename: path to file\n");
		printf("\talgorithm: encryption algorithm to be used. Can be either a number or a string\n");
		printf("\t    rot_13 => 'rot_13' or %d\n\n", ROT_13);
		printf("\t-d: decrypt file that was encrypted\n");
		exit(0);
	}
	filename = basename(argv[1]);
	f = fopen(argv[1], "r");
	if (f == NULL) {
		printf("Couldn't find file '%s'. Are you sure you typed the right path?\n", argv[1]);
		exit(1);
	}
	if (strcmp(argv[2], "rot_13") == 0 || strcmp(argv[2], "0") == 0) {
		algorithm = ROT_13;
	}

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);

	char *string = malloc(size + 1);
	fread(string, size, 1, f);
	string[size] = '\0';
	fclose(f);

	strcpy(enc_path, ENCRYPTED_FILES_DIR);
	strcat(enc_path, filename);
	if (stat(ENCRYPTED_FILES_DIR, &st) == -1) 
    	mkdir(ENCRYPTED_FILES_DIR, 0777);
	f = fopen(enc_path, "w");
	if (f == NULL) {
		printf("Can't write to file\n");
		exit(1);
	}

	switch(algorithm) {
		case ROT_13:
			rot13_encrypt(string);
			fprintf(f, "%s", string);
			break;
		default:
			printf("I don't know that algorithm...\n");
			break;
	}
	fclose(f);

	if (argc >= 4 && strcmp(argv[3], "-d") == 0) {
		strcpy(enc_path, DECRYPTED_FILES_DIR);
		strcat(enc_path, filename);
		if (stat(DECRYPTED_FILES_DIR, &st) == -1) 
    		mkdir(DECRYPTED_FILES_DIR, 0777);
		f = fopen(enc_path, "w");
		if (f == NULL) {
			printf("Can't write to file\n");
			exit(1);
		}
	}

	switch(algorithm) {
		case ROT_13:
			rot13_decrypt(string);
			fprintf(f, "%s", string);
			break;
		default:
			printf("I don't know that algorithm...\n");
			break;
	}
	fclose(f);

	free(string);
	return 0;
}
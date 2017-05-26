#include "base64.cuh"
#include "rot-13.cuh"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <libgen.h>

#define ROT_13 0
#define BASE_64 1
#define ENCODED_FILES_DIR "encoded_files/"
#define DECODED_FILES_DIR "decoded_files/"

int main(int argc, char *argv[]) {
	FILE *f;
	char *filename;
	char enc_path[1000];
	long size;
	int algorithm = -1;
	BYTE *out;
	size_t out_length;
	struct stat st = {0};
	if (argc <= 2) {
		printf("usage: %s [filename] [algorithm] -d\n", argv[0]);
		printf("\tfilename: path to file\n");
		printf("\talgorithm: encoding algorithm to be used. Can be either a number or a string\n");
		printf("\t    rot_13 => 'rot_13' or %d\n\n", ROT_13);
		printf("\t    base64 => 'base64' or %d\n\n", ROT_13);
		printf("\t-d: decode file after encoding it\n");
		exit(0);
	}
	filename = basename(argv[1]);
	f = fopen(argv[1], "rb");
	if (f == NULL) {
		printf("Couldn't find file '%s'. Are you sure you typed the right path?\n", argv[1]);
		exit(1);
	}
	if (strcmp(argv[2], "rot_13") == 0 || strcmp(argv[2], "0") == 0) {
		algorithm = ROT_13;
	} else if (strcmp(argv[2], "base64") == 0 || strcmp(argv[2], "1") == 0) {
		algorithm = BASE_64;
	}

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);

	unsigned char *string = malloc((size + 1) * sizeof(char));
	fread(string, size, 1, f);
	if (algorithm == ROT_13)
		string[size] = '\0';
	fclose(f);

	strcpy(enc_path, ENCODED_FILES_DIR);
	if (stat(enc_path, &st) == -1) 
    	mkdir(enc_path, 0777);
	strcat(enc_path, argv[2]);
	strcat(enc_path, "/");
	if (stat(enc_path, &st) == -1) 
    	mkdir(enc_path, 0777);
   	strcat(enc_path, filename);
   	
	f = fopen(enc_path, "wb");
	if (f == NULL) {
		printf("Can't write to file\n");
		exit(1);
	}

	switch(algorithm) {
		case ROT_13:
			rot13_encode(string, size);
			fprintf(f, "%s", string);
			break;
		case BASE_64:
			out_length = size / 3 * 4 + size / 57;
			if (size % 3) out_length += 4;
			out = malloc(out_length * sizeof(BYTE));
			base64_encode(string, size, out, out_length);
			fprintf(f, "%s", out);
			break;
		default:
			printf("I don't know that algorithm...\n");
			break;
	}
	fclose(f);

	if (argc >= 4 && strcmp(argv[3], "-d") == 0) {
		strcpy(enc_path, DECODED_FILES_DIR);
		if (stat(enc_path, &st) == -1) 
    		mkdir(enc_path, 0777);
		strcat(enc_path, argv[2]);
		strcat(enc_path, "/");
		if (stat(enc_path, &st) == -1) 
    		mkdir(enc_path, 0777);
		strcat(enc_path, filename);

		f = fopen(enc_path, "wb");
		if (f == NULL) {
			printf("Can't write to file\n");
			exit(1);
		}

		switch(algorithm) {
			case ROT_13:
				rot13_decode(string, size);
				fprintf(f, "%s", string);
				break;
			case BASE_64:
				base64_decode(out, out_length, string, size);
				fprintf(f, "%s", string);
				break;
			default:
				printf("I don't know that algorithm...\n");
				break;
		}

		fclose(f);
	}

	free(string);
	free(out);
	return 0;
}
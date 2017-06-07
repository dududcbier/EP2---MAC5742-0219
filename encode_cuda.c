#include "base64.cuh"
#include "rot-13.cuh"
#include "arcfour.cuh"

#include "seq-rot-13.h"
#include "seq-base64.h"
#include "arcfour.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <libgen.h>

#define CUDA_ROT13 1
#define CUDA_BASE64 2
#define CUDA_RC4 3
#define SEQ_ROT13 -1
#define SEQ_BASE64 -2
#define SEQ_RC4 -3
#define ENCODED_FILES_DIR "encoded_files/"
#define DECODED_FILES_DIR "decoded_files/"
#define DEFAULT_KEY_LEN 10
#define DEFAULT_KEY "secret key"

void printUsage(char program_name[]);
void createPath(char type[], int sequential, char *algorithm);

char enc_path[1000];

int main(int argc, char *argv[]) {
	FILE *f;
	char *filename;
	long size;
	int algorithm = 0, decode = 0, silent = 0;
	BYTE *out = NULL;
	BYTE *key = NULL;
	size_t out_length;
	if (argc <= 2) {
		printUsage(argv[0]);
		exit(0);
	}
	filename = basename(argv[1]);
	f = fopen(argv[1], "rb");
	if (f == NULL) {
		if (!silent) printf("Couldn't find file '%s'. Are you sure you typed the right path?\n", argv[1]);
		exit(1);
	}

	if (strcmp(argv[2], "rot_13") == 0 || strcmp(argv[2], "1") == 0) {
		algorithm = CUDA_ROT13;
	} else if (strcmp(argv[2], "base64") == 0 || strcmp(argv[2], "2") == 0) {
		algorithm = CUDA_BASE64;
	} else if (strcmp(argv[2], "arcfour") == 0 || strcmp(argv[2], "3") == 0) {
		algorithm = CUDA_RC4;
		key = malloc(DEFAULT_KEY_LEN * sizeof(BYTE));
		for (int i = 0; i < DEFAULT_KEY_LEN; i++)
			key[i] = DEFAULT_KEY[i];
	}

	for (int i = 3; i <= argc - 1 && argv[i][0] == '-'; i++) 
		switch(argv[i][1]) {
			case 'd': decode = 1; break;
			case 's': algorithm *= -1; break;
			case 'x': silent = 1; break;
			default:
				printUsage(argv[0]);
				exit(0);
				break;
		}

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);

	unsigned char *string = malloc(size * sizeof(char));
	if (fread(string, sizeof(BYTE), size, f) != size)
		printf("Something went wrong reading file!");
	fclose(f);

	createPath(ENCODED_FILES_DIR, algorithm < 0, argv[2]);
   	strcat(enc_path, filename);

	f = fopen(enc_path, "wb");
	if (f == NULL) {
		if (!silent) printf("Can't write to file\n");
		exit(1);
	}
	switch(algorithm) {
		case CUDA_ROT13:
			rot13_encode(string, size);
			fprintf(f, "%s", string);
			break;
		case CUDA_BASE64:
			out_length = size / 3 * 4 + size / 57;
			if (size % 3) out_length += 4;
			out = malloc(out_length * sizeof(BYTE));
			base64_encode(string, size, out, out_length);
			fwrite(out, sizeof(BYTE), out_length, f);
			break;
		case CUDA_RC4:
			arcfour_encode(string, size, key, DEFAULT_KEY_LEN);
			fwrite(string, sizeof(BYTE), size, f);
			break;
		case SEQ_ROT13:
			rot13((char *)string);
			fprintf(f, "%s", string);
			break;
		case SEQ_BASE64:
			out_length = seq_base64_encode(string, NULL, size, 1);
			out = malloc(out_length * sizeof(BYTE));
			seq_base64_encode(string, out, size, 1);
			fwrite(out, sizeof(BYTE), out_length, f);
			break;
		case SEQ_RC4:
			arcfour(string, size, key, DEFAULT_KEY_LEN);
			fwrite(string, sizeof(BYTE), size, f);
			break;
		default:
			if (!silent) printf("I don't know that algorithm...\n");
			break;
	}
	fclose(f);

	if (decode) {
		createPath(DECODED_FILES_DIR, algorithm < 0, argv[2]);
   		strcat(enc_path, filename);

		f = fopen(enc_path, "wb");
		if (f == NULL) {
			if (!silent) printf("Can't write to file\n");
			exit(1);
		}
		switch(algorithm) {
			case CUDA_ROT13:
				rot13_decode(string, size);
				fprintf(f, "%s", string);
				break;
			case CUDA_BASE64:
				base64_decode(out, out_length, string, size);
				fwrite(string, sizeof(BYTE), size, f);
				break;
			case CUDA_RC4:
				arcfour_decode(string, size, key, DEFAULT_KEY_LEN);
				fwrite(string, sizeof(BYTE), size, f);
				break;
			case SEQ_ROT13:
				rot13((char *)string);
				fprintf(f, "%s", string);
				break;
			case SEQ_BASE64:
				seq_base64_decode(out, string, out_length);
				fwrite(string, sizeof(BYTE), size, f);
				break;
			case SEQ_RC4:
				arcfour(string, size, key, DEFAULT_KEY_LEN);
				fwrite(string, sizeof(BYTE), size, f);
				break;
			default:
				if (!silent) printf("I don't know that algorithm...\n");
				break;
		}
		fclose(f);
	}

	free(string);
	if (out != NULL) free(out);
	if (key != NULL) free(key);
	return 0;
}

void printUsage(char program_name[]) {
	printf("usage: %s filename algorithm [-d] [-s] [-x]\n", program_name);
	printf("\tfilename: path to file\n");
	printf("\talgorithm: encoding algorithm to be used. Can be either a number or a string\n");
	printf("\t    rot_13 => 'rot_13' or %d\n\n", CUDA_ROT13);
	printf("\t    base64 => 'base64' or %d\n\n", CUDA_BASE64);
	printf("\t    arcfour => 'arcfour' or %d\n\n", CUDA_RC4);
	printf("\t-d: decode file after encoding it\n");
	printf("\t-s: use sequential algorithm\n");
	printf("\t-x: silent mode\n");
}

void createPath(char type[], int sequential, char *algorithm) {
	struct stat st = {0};
	strcpy(enc_path, type);
	if (stat(enc_path, &st) == -1) 
    	mkdir(enc_path, 0777);
    if (sequential){
    	strcat(enc_path, "sequential/");
    	if (stat(enc_path, &st) == -1) 
    		mkdir(enc_path, 0777);
    }
	strcat(enc_path, algorithm);
	strcat(enc_path, "/");
	if (stat(enc_path, &st) == -1) 
    	mkdir(enc_path, 0777);
}
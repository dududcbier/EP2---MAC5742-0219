#ifndef ARCFOUR
#define ARCFOUR

void arcfour_encode(unsigned char *data, unsigned long length, unsigned char *key, unsigned long key_length);
void arcfour_decode(unsigned char *data, unsigned long length, unsigned char *key, unsigned long key_length);
#endif
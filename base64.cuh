#ifndef BASE64
#define BASE64

void base64_encode(unsigned char *data, unsigned long length, unsigned char *output, unsigned long out_length);
void base64_decode(unsigned char *data, unsigned long length, unsigned char *output, unsigned long out_length);
#endif
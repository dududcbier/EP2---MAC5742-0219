#ifndef BASE64
#define BASE64

void blowfish_decrypt(unsigned char *in, unsigned char *out, BLOWFISH_KEY *keystruct);
void blowfish_encrypt(unsigned char *in, unsigned char *out, BLOWFISH_KEY *keystruct);

#endif

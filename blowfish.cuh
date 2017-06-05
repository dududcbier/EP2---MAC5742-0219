#ifndef BASE64
#define BASE64

void blowfish_decrypt(const unsigned char *in, unsigned char *out, const BLOWFISH_KEY *keystruct);
void blowfish_encrypt(const unsigned char *in, unsigned char *out, const BLOWFISH_KEY *keystruct);

#endif

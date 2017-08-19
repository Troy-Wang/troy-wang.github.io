---
layout:     post
title:      "Methods for Encryption in Python"
subtitle:   "Python中一些加解密方法"
date:       2017-08-18
author:     "Troy Wang"
header-img: "img/post/default-dusk.jpg"
tags:
    - Python
---

## 1. Base64

```python
import base64

str = 'hello world'
ret1 = base64.encodestring(str)
ret2 = base64.decodestring(ret1)
print ret1, ret2

```

## 2. AES

```python
# -*- coding: utf-8 -*-
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex


class prpcrypt():
    def __init__(self, key):
        self.key = self.pad(key)
        self.mode = AES.MODE_CBC

    # length of key must be 16(AES-128), 24（AES-192), or 32(AES-256) Bytes
    def pad(self, key):
        if len(key) > 16:
            return key[0:16]
        else:
            return key.rjust(16)

    # length of the text to be encrypted must be a multiple of 16
    def encrypt(self, text):
        cryptor = AES.new(self.key, self.mode, self.key)
        length = 16
        add = length - (len(text) % length)
        text = text + ('\0' * add)
        self.ciphertext = cryptor.encrypt(text)
        # result of the AES encryption may not be ascii, need to be converted to hex
        return b2a_hex(self.ciphertext)

    # after decryption，use strip() to remove the padding
    def decrypt(self, text):
        cryptor = AES.new(self.key, self.mode, self.key)
        plain_text = cryptor.decrypt(a2b_hex(text))
        return plain_text.rstrip('\0')


if __name__ == '__main__':
    pc = prpcrypt('keyskeyskeyskey')
    e = pc.encrypt("1234567890")
    d = pc.decrypt(e)
    print e, d
    e = pc.encrypt("123456789012345678901234567890")
    d = pc.decrypt(e)
    print e, d

```

## 3. RSA

```python
# -*- coding: utf-8 -*-
import rsa

# generate new public & private key
(publicKey, privateKey) = rsa.newkeys(1024)

# write public & private key to pem file
with open('public.pem', 'w+') as f:
    f.write(publicKey.save_pkcs1().decode())

with open('private.pem', 'w+') as f:
    f.write(privateKey.save_pkcs1().decode())

# read public & private key from pem file
with open('public.pem', 'r') as f:
    pubKey = rsa.PublicKey.load_pkcs1(f.read().encode())

with open('private.pem', 'r') as f:
    pvtKey = rsa.PrivateKey.load_pkcs1(f.read().encode())

message = '1234567890'
# encrypt with public key
encryptRet = rsa.encrypt(message.encode(), pubKey)
print encryptRet
# decrypt with private key
decryptRet = rsa.decrypt(encryptRet, pvtKey).decode()
print decryptRet

```

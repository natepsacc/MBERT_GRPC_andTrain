#!/bin/bash

# Setup keys
openssl genrsa -out ca.key 4096

openssl req -new -x509 -key ca.key -out ca.crt -days 3650 -subj "/CN=MyCA"

openssl genrsa -out server.key 4096

openssl req -new -key server.key -out server.csr -subj "/CN=localhost"

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365

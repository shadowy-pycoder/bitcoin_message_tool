Bitcoin Message Tool
======

Bitcoin message signing/verification tool

A lightweight CLI tool for signing and verification of bitcoin messages.
Bitcoin message is the most straightforward and natural way to prove ownership over
a given address without revealing any confidential information.

Installation
------------

To install with pip, run:

    pip install bitcoin-message-tool

Quickstart Guide
----------------

    Usage:

    python -m bitcoin_message_tool -h

    or

    python bmt.py -h
    usage: python3 bmt.py [-h] {sign,verify} ...

    Bitcoin message signing/verification tool

    positional arguments:
    {sign,verify}

    options:
    -h, --help     show this help message and exit

Message signing:

    python bmt.py sign -h
    usage: python3 bmt.py sign [-h] -p -a {p2pkh,p2wpkh-p2sh,p2wpkh} -m [MESSAGE ...] [-d] [-v]

    options:
    -h, --help            show this help message and exit

    Sign messsage:
    -p, --privkey         private key in wallet import format (WIF)
    -a {p2pkh,p2wpkh-p2sh,p2wpkh}, --addr_type {p2pkh,p2wpkh-p2sh,p2wpkh}
                            type of bitcoin address
    -m [MESSAGE ...], --message [MESSAGE ...]
                            Message to sign
    -d, --deterministic   sign deterministtically (RFC6979)
    -v, --verbose         print prettified message

Example 1:
Non-deterministic signature for compressed private key and p2pkh address

    $python bmt.py sign -p -a p2pkh -m ECDSA is the most fun I have ever experienced

    PrivateKey(WIF): <insert private key here>

Output:

    Bitcoin address: 175A5YsPUdM71mnNCC3i8faxxYJgBonjWL
    Message: ECDSA is the most fun I have ever experienced
    Signature: IBuc5GXSJCr6m7KevsBAoCiX8ToOjW2CDZMr6PCEbiHwQJ237LZTj/REbDHI1/yelY6uBWEWXiOWoGnajlgvO/A=

Example 2:
Deterministic signature for compressed private key and p2pkh address

    $python bmt.py sign -p -a p2pkh -m ECDSA is the most fun I have ever experienced -d

    PrivateKey(WIF): <insert private key here>

Output:

    Bitcoin address: 175A5YsPUdM71mnNCC3i8faxxYJgBonjWL
    Message: ECDSA is the most fun I have ever experienced
    Signature: HyiLDcQQ1p2bKmyqM0e5oIBQtKSZds4kJQ+VbZWpr0kYA6Qkam2MlUeTr+lm1teUGHuLapfa43JjyrRqdSA0pxs=

Example 3:
Deterministic signature for compressed private key and p2pkh address (verbose mode)

    $python bmt.py sign -p -a p2pkh -m ECDSA is the most fun I have ever experienced -d -v

    PrivateKey(WIF): <insert private key here>

Output:

    -----BEGIN BITCOIN SIGNED MESSAGE-----
    ECDSA is the most fun I have ever experienced
    -----BEGIN BITCOIN SIGNATURE-----
    175A5YsPUdM71mnNCC3i8faxxYJgBonjWL

    HyiLDcQQ1p2bKmyqM0e5oIBQtKSZds4kJQ+VbZWpr0kYA6Qkam2MlUeTr+lm1teUGHuLapfa43JjyrRqdSA0pxs=
    -----END BITCOIN SIGNATURE-----

Example 4:
Uncompressed private keys can't produce addresses other than 'p2pkh'

    python bmt.py sign -p -m ECDSA is the most fun I have ever experienced -a 'p2wpkh'  -d -v

    PrivateKey(WIF): <insert private key here>

Output:

    Traceback (most recent call last):
    ...
    PrivateKeyError: ('Need WIF-compressed private key for this address type:', 'p2wpkh')

Message verification:

    python bmt.py verify -h
    usage: python3 bmt.py verify [-h] -a ADDRESS -m [MESSAGE ...] -s SIGNATURE [-v] [-r]

    options:
    -h, --help            show this help message and exit

    Verify messsage:
    -a ADDRESS, --address ADDRESS
                            specify bitcoin address
    -m [MESSAGE ...], --message [MESSAGE ...]
                            Message to verify
    -s SIGNATURE, --signature SIGNATURE
                            bitcoin signature in base64 format
    -v, --verbose         print full message
    -r, --recpub          recover public key

Example 1:
Standard message verification

    python bmt.py verify -a 175A5YsPUdM71mnNCC3i8faxxYJgBonjWL \
    > -m ECDSA is the most fun I have ever experienced \
    > -s HyiLDcQQ1p2bKmyqM0e5oIBQtKSZds4kJQ+VbZWpr0kYA6Qkam2MlUeTr+lm1teUGHuLapfa43JjyrRqdSA0pxs=

Output:

    True

Example 2:
Message verification in verbose mode

    python bmt.py verify -a 175A5YsPUdM71mnNCC3i8faxxYJgBonjWL \
    > -m ECDSA is the most fun I have ever experienced \
    > -s HyiLDcQQ1p2bKmyqM0e5oIBQtKSZds4kJQ+VbZWpr0kYA6Qkam2MlUeTr+lm1teUGHuLapfa43JjyrRqdSA0pxs= \
    > -v

Output:

    True
    Message verified to be from 175A5YsPUdM71mnNCC3i8faxxYJgBonjWL

Example 3:
Display a recovered public key

    python bmt.py verify -a 175A5YsPUdM71mnNCC3i8faxxYJgBonjWL \
    > -m ECDSA is the most fun I have ever experienced \
    > -s HyiLDcQQ1p2bKmyqM0e5oIBQtKSZds4kJQ+VbZWpr0kYA6Qkam2MlUeTr+lm1teUGHuLapfa43JjyrRqdSA0pxs= \
    > --recpub

Output:

    True
    024aeaf55040fa16de37303d13ca1dde85f4ca9baa36e2963a27a1c0c1165fe2b1

Example 4:
Error message

    python bmt.py verify -a 175A5YsPUdM71mnNCC3i8faxxYJgBonjWL \
    > -m ECDSA is the most fun I have ever experienced \
    > -s HyiLDcQQ1p2bKmyqM0e5oIBQtKSZds4kJQ+VbZWpr0kYA6Qkam2MlUeTr+lm1teUGHuLaffa43Jj= -v -r \

Output:

    Traceback (most recent call last):
    ...
    SignatureError: ('Signature must be 65 bytes long:', 57)

Contribute
----------

If you'd like to contribute to bitcoin_message_signer, check out https://github.com/shadowy_pycoder/bitcoin_message_tool

"""
Bitcoin Message Tool by shadowy-pycoder 30.01.2023 https://github.com/shadowy-pycoder/

A lightweight CLI tool for signing and verification of bitcoin messages.
Bitcoin message is the most straightforward and natural way to prove ownership over
a given address without revealing any confidential information.

This tool closely follows specification described in BIP137:

Please note that "since this format includes P2PKH keys, it is backwards compatible, 
but keep in mind some software has checks for ranges of headers and will report the newer 
segwit header types as errors."

More info: https://github.com/bitcoin/bips/blob/master/bip-0137.mediawiki




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

Message signing

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

Message verification

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


"""

import argparse
import base64
import getpass
import hmac
import secrets
import sys
from hashlib import sha256
from typing import NamedTuple

import base58
import bech32  # type: ignore
from ripemd.ripemd160 import ripemd160  # type: ignore

# ################################ non-public interface (see below) #################################


class Point(NamedTuple):
    """Point on elliptic curve"""
    x: int
    y: int


class JacobianPoint(NamedTuple):
    """Point on elliptic curve in jacobian coordinates"""
    # https://en.wikipedia.org/wiki/Jacobian_curve
    x: int
    y: int
    z: int


class EllipticCurve(NamedTuple):
    """Elliptic curve with all the parameters to define it."""
    p_curve: int
    n_curve: int
    a_curve: int
    b_curve: int
    gen_point: JacobianPoint


class Signature(NamedTuple):
    """Elliptic curve digital signature"""
    # https://en.wikipedia.org/wiki/Elliptic_Curve_Digital_Signature_Algorithm
    r: int
    s: int


class BitcoinMessageError(Exception):
    """Base exception for Bitcoin Message Tool"""


class PrivateKeyError(BitcoinMessageError):
    """Private key is out of allowed range"""


class PointError(BitcoinMessageError):
    """Point is not on elliptic curve"""


class SignatureError(BitcoinMessageError):
    """Invalid ECDSA signature parameters"""


class PrivateKey(argparse.Action):
    """Make private keys not appear in CLI"""
    # https://stackoverflow.com/questions/29948567/how-to-suppress-the-display-of-passwords/29948740#29948740

    def __call__(self, parser, namespace, values, option_string):
        values = getpass.getpass(prompt='PrivateKey(WIF): ')
        setattr(namespace, self.dest, values)


P_CURVE = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N_CURVE = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
A_CURVE = 0
B_CURVE = 7
GEN_POINT = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
             0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
IDENTITY_POINT = JacobianPoint(x=P_CURVE, y=0, z=1)
POW_2_256_M1 = 2 ** 256 - 1

secp256k1 = EllipticCurve(p_curve=P_CURVE, n_curve=N_CURVE,
                          a_curve=A_CURVE, b_curve=B_CURVE,
                          gen_point=JacobianPoint(x=GEN_POINT[0], y=GEN_POINT[1], z=1))


precomputes: list[JacobianPoint] = []
headers = [[b'\x1b', b'\x1c', b'\x1d', b'\x1e'],
           [b'\x1f', b'\x1f', b'\x20', b'\x22'],
           [b'\x23', b'\x24', b'\x25', b'\x26'],
           [b'\x27', b'\x29', b'\x28', b'\x2a'],
           [b'\x2b', b'\x2c', b'\x2d', b'\x2e']]


def double_sha256(b: bytes) -> bytes:
    return sha256(sha256(b).digest()).digest()


def ripemd160_sha256(b: bytes) -> bytes:
    return ripemd160(sha256(b).digest())


def is_odd(n: int) -> int:
    return n & 1


def mod_inverse(n: int, /, mod: int) -> int:
    return pow(n, -1, mod)


def generate() -> int:
    """Generate cryptographically-secure random integer"""
    return secrets.randbelow(secp256k1.n_curve)


def to_affine(p: JacobianPoint, /) -> Point:
    """Convert jacobian point to affine point"""
    inv_z = mod_inverse(p.z, secp256k1.p_curve)
    inv_z2 = pow(inv_z, 2)
    x = (p.x * inv_z2) % secp256k1.p_curve
    y = (p.y * inv_z2 * inv_z) % secp256k1.p_curve
    return Point(x, y)


def to_jacobian(p: Point, /) -> JacobianPoint:
    """Convert affine point to jacobian point"""
    return JacobianPoint(p.x, p.y, z=1)


def valid_point(p: Point | tuple[int, int], /) -> bool:
    """Check if a given point belongs to secp256k1 elliptic curve"""
    try:
        return (all(isinstance(i, int) for i in p) and
                pow(p[1], 2) % secp256k1.p_curve == (pow(p[0], 3) + secp256k1.b_curve) % secp256k1.p_curve)
    except (TypeError, IndexError):  # Exception is raised when given arguments are invalid (non-integers)
        return False  # which also means point is not on curve


def valid_key(scalar: int, /) -> bool:
    """Check if an integer is within allowed range"""
    return isinstance(scalar, int) and not (scalar <= 0 or scalar >= secp256k1.n_curve)


def valid_checksum(version: bytes, privkey: bytes, checksum: bytes, /) -> bool:
    return double_sha256(version + privkey)[:4] == checksum


def to_bytes(wif: str, /) -> tuple[bytes, bytes, bytes]:
    """Convert WIF private key to bytes"""
    if not isinstance(wif, str):
        raise PrivateKeyError('must be in WIF format')
    privkey = base58.b58decode(wif)
    return privkey[:1], privkey[1:-4], privkey[-4:]


def to_int(wif: str, /) -> tuple[int, bool]:
    """Convert WIF private key to integer"""
    if not isinstance(wif, str):
        raise PrivateKeyError('must be in WIF format')
    try:
        version, privkey, checksum = to_bytes(wif)
    except ValueError:
        raise PrivateKeyError('Invalid scalar/private key')
    if not valid_checksum(version, privkey, checksum):
        raise PrivateKeyError('invalid WIF checksum')
    if len(privkey) == 33:
        privkey_int = int.from_bytes(privkey[:-1], 'big')
        uncompressed = False
    else:
        privkey_int = int.from_bytes(privkey, 'big')
        uncompressed = True
    if valid_key(privkey_int):
        return privkey_int, uncompressed
    raise PrivateKeyError('Invalid scalar/private key')


def ec_dbl(q: JacobianPoint, /) -> JacobianPoint:
    # Fast Prime Field Elliptic Curve Cryptography with 256 Bit Primes
    # Shay Gueron, Vlad Krasnov
    # https://eprint.iacr.org/2013/816.pdf page 4
    if q.x == secp256k1.p_curve:
        return q
    Y2 = q.y * q.y
    S = (4 * q.x * Y2) % secp256k1.p_curve
    M = 3 * q.x * q.x
    x = (M * M - 2 * S) % secp256k1.p_curve
    y = (M * (S - x) - 8 * Y2 * Y2) % secp256k1.p_curve
    z = (2 * q.y * q.z) % secp256k1.p_curve
    return JacobianPoint(x, y, z)


def ec_add(p: JacobianPoint, q: JacobianPoint, /) -> JacobianPoint:
    # Fast Prime Field Elliptic Curve Cryptography with 256 Bit Primes
    # Shay Gueron, Vlad Krasnov
    # https://eprint.iacr.org/2013/816.pdf page 4
    if p.x == secp256k1.p_curve:
        return q
    if q.x == secp256k1.p_curve:
        return p

    PZ2 = p.z * p.z
    QZ2 = q.z * q.z
    U1 = (p.x * QZ2) % secp256k1.p_curve
    U2 = (q.x * PZ2) % secp256k1.p_curve
    S1 = (p.y * QZ2 * q.z) % secp256k1.p_curve
    S2 = (q.y * PZ2 * p.z) % secp256k1.p_curve

    if U1 == U2:
        if S1 == S2:  # double point
            return ec_dbl(p)
        else:  # return POINT_AT_INFINITY
            return IDENTITY_POINT

    H = (U2 - U1) % secp256k1.p_curve
    R = (S2 - S1) % secp256k1.p_curve
    H2 = (H * H) % secp256k1.p_curve
    H3 = (H2 * H) % secp256k1.p_curve
    x = (R * R - H3 - 2 * U1 * H2) % secp256k1.p_curve
    y = (R * (U1 * H2 - x) - S1 * H3) % secp256k1.p_curve
    z = (H * p.z * q.z) % secp256k1.p_curve
    return JacobianPoint(x, y, z)


def get_precomputes() -> None:
    dbl: JacobianPoint = secp256k1.gen_point
    for _ in range(256):
        precomputes.append(dbl)
        dbl = ec_dbl(dbl)


def ec_mul(scalar: int, point: Point | JacobianPoint | None = None, /) -> JacobianPoint:
    # https://paulmillr.com/posts/noble-secp256k1-fast-ecc/#fighting-timing-attacks
    n = scalar
    p = IDENTITY_POINT
    if point is None:  # no point specified, which means standard multiplication
        fake_p = p
        fake_n = POW_2_256_M1 ^ n
        if not precomputes:
            get_precomputes()
        for precomp in precomputes:
            q = precomp
            if n & 1:
                p = ec_add(p, q)
            else:
                fake_p = ec_add(fake_p, q)
            n >>= 1
            fake_n >>= 1
    else:  # unsafe multiplication for signature verification
        if isinstance(point, Point):
            point = to_jacobian(point)
        q = point
        while n > 0:
            if n & 1:
                p = ec_add(p, q)
            n >>= 1
            q = ec_dbl(q)
    return JacobianPoint(p.x, p.y, p.z)


def create_raw_pubkey(privkey) -> Point:
    raw_pubkey = to_affine(ec_mul(privkey))
    if not valid_point(raw_pubkey):
        raise PointError('Point is not on curve')
    return raw_pubkey


def create_pubkey(raw_pubkey: Point, /, *, uncompressed: bool = False) -> bytes:
    if uncompressed:
        return b'\x04' + raw_pubkey.x.to_bytes(32, 'big') + raw_pubkey.y.to_bytes(32, 'big')
    prefix = b'\x03' if is_odd(raw_pubkey.y) else b'\x02'
    return prefix + raw_pubkey.x.to_bytes(32, 'big')


def create_address(pubkey: bytes, /) -> str:
    address = b'\x00' + ripemd160_sha256(pubkey)
    return base58.b58encode_check(address).decode('UTF-8')


def create_nested_segwit(pubkey: bytes, /) -> str:
    address = b'\x05' + ripemd160_sha256(b'\x00\x14' + ripemd160_sha256(pubkey))
    return base58.b58encode_check(address).decode('UTF-8')


def create_native_segwit(pubkey: bytes, /) -> str:
    return bech32.encode('bc', 0x00, ripemd160_sha256(pubkey))


def varint(length: int) -> bytes:
    # https://en.bitcoin.it/wiki/Protocol_documentation#Variable_length_integer
    if length < 0xFD:
        return length.to_bytes(1, 'little')
    elif length <= 0xFFFF:
        return b'\xFD' + length.to_bytes(2, 'little')
    elif length <= 0xFFFFFFFF:
        return b'\xFE' + length.to_bytes(4, 'little')
    elif length <= 0xFFFFFFFFFFFFFFFF:
        return b'\xFF' + length.to_bytes(8, 'little')
    else:
        raise SignatureError(f'Message is too lengthy: {length}')


def msg_magic(msg: str) -> bytes:
    message = msg.encode('utf-8')
    return b'\x18Bitcoin Signed Message:\n' + varint(len(message)) + message


def signed(privkey: int, msg: int, k: int) -> Signature | None:
    """
    Calculate r and s values of a signature


    Just a helper function that should not be used directly

    Args:

    privkey - random integer

    msg - hash of a signature (usually double sha256 of a message with 'msg magic' applied)

    k - nonce that comes from random (SystemRandom) or pseudorandom source (RFC6979)
    """
    if not valid_key(k):
        return None
    # when working with private keys, standard multiplication is used
    point = to_affine(ec_mul(k))
    r = point.x % secp256k1.n_curve
    if r == 0 or point == IDENTITY_POINT:
        return None
    s = mod_inverse(k, secp256k1.n_curve) * (msg + privkey * r) % secp256k1.n_curve
    if s == 0:
        return None
    if s > secp256k1.n_curve >> 1:  # https://github.com/bitcoin/bips/blob/master/bip-0062.mediawiki
        s = secp256k1.n_curve - s
    return Signature(r, s)


def bits_to_int(b: bytes, qlen: int):
    # https://www.rfc-editor.org/rfc/rfc6979 section 2.3.2.
    blen = len(b) << 3
    b_int = int.from_bytes(b, 'big')
    if blen > qlen:
        b_int = b_int >> blen - qlen
    return b_int


def int_to_oct(x: int, rolen: int) -> bytes:
    # https://www.rfc-editor.org/rfc/rfc6979 section 2.3.3.
    xolen = x.bit_length() >> 3
    x_hex = f'{x:x}'
    if xolen < rolen:
        x_hex = f'{x:0>{rolen << 1}x}'
    elif xolen > rolen:
        x_hex = x_hex[xolen - rolen << 1:]
    return bytes.fromhex(x_hex)


def bits_to_oct(b: bytes, q: int, qlen: int, rolen: int) -> bytes:
    # https://www.rfc-editor.org/rfc/rfc6979 section 2.3.4.
    z1 = bits_to_int(b, qlen)
    z2 = z1 - q
    if z2 < 0:
        z2 = z1
    return int_to_oct(z2, rolen)


def rfc_sign(x: int, msg: int, q: int) -> Signature:
    # https://www.rfc-editor.org/rfc/rfc6979 section 3.2.
    qlen = q.bit_length()
    qolen = qlen >> 3
    rolen = qlen + 7 >> 3
    # step a is omitted since we already have a hash of a message
    h1 = msg.to_bytes(32, 'big')
    # step b
    V = b'\x01' * 32
    # step c
    K = b'\x00' * 32
    # step d
    m1 = b'\x00' + int_to_oct(x, rolen) + bits_to_oct(h1, q, qlen, rolen)
    m2 = b'\x01' + int_to_oct(x, rolen) + bits_to_oct(h1, q, qlen, rolen)
    K_ = hmac.new(K, digestmod=sha256)
    K_.update(V + m1)
    K = K_.digest()
    # step e
    V = hmac.new(K, V, digestmod=sha256).digest()
    # step f
    K_ = hmac.new(K, digestmod=sha256)
    K_.update(V + m2)
    K = K_.digest()
    # step g
    V = hmac.new(K, V, digestmod=sha256).digest()
    # step h
    while True:
        T = b''
        while len(T) < qolen:
            V = hmac.new(K, V, digestmod=sha256).digest()
            T = T + V
        k = bits_to_int(T, qlen)
        if (sig := signed(x, msg, k)) is not None:
            return sig
        # if k was invalid (sig is None), continue with algorithm
        K_ = hmac.new(K, digestmod=sha256)
        K_.update(V + b'\x00')
        K = K_.digest()
        V = hmac.new(K, V, digestmod=sha256).digest()


def sign(privkey: int, msg: int, /) -> Signature:
    # https://learnmeabitcoin.com/technical/ecdsa#sign
    while True:
        k = generate()
        if (sig := signed(privkey, msg, k)) is not None:
            return sig


def derive_address(pubkey: bytes, addr_type: str) -> tuple[str, int]:
    """
    Derive bitcoin address from public key and given address type


    This function shouldn't be called directly

    Args:

    pubkey - bitcoin public key in bytes format

    addr_type - either 'p2pkh', 'p2wpkh-p2sh' or 'p2wpkh'
    """

    if pubkey.startswith(b'\x04') and addr_type != 'p2pkh':
        raise PrivateKeyError('Need WIF-compressed private key for this address type:', addr_type)
    elif pubkey.startswith(b'\x04'):
        return create_address(pubkey), 0
    elif addr_type.lower() == 'p2pkh':
        return create_address(pubkey), 1
    elif addr_type.lower() == 'p2wpkh-p2sh':
        return create_nested_segwit(pubkey), 2
    elif addr_type.lower() == 'p2wpkh':
        return create_native_segwit(pubkey), 3
    else:
        raise SignatureError('Invalid address type')

# ################################ public interface starts here #################################


def sign_message(wif: str, addr_type: str, message: str, /, *, deterministic=False) -> tuple[str, ...]:
    """
    Sign message with private key (WIF) and specified address type

    Args:

    wif - private key in compressed or uncompressed format. 
    Compressed private key will produce compressed public key and address.
    Uncompressed private key will only produce one address type - uncompressed legacy address

    addr_type - specify which address type you want to use to produce signature. 
    It can be either p2pkh (compressed and uncompressed), p2wpkh-p2sh or p2wpkh (only compressed).

    message - string that will be used for signing

    deterministic - if you want your signatures be produced deterministically, set this flag to True,
    and each unique combination of private key and message will yield only one signature 

    """
    m_bytes = msg_magic(message)
    msg = int.from_bytes(double_sha256(m_bytes), 'big')
    privkey, uncompressed = to_int(wif)
    raw_pubkey = create_raw_pubkey(privkey)
    pubkey = create_pubkey(raw_pubkey, uncompressed=uncompressed)
    if not deterministic:
        sig = sign(privkey, msg)
    else:
        sig = rfc_sign(privkey, msg, secp256k1.n_curve)
    address, ver = derive_address(pubkey, addr_type)
    r = sig.r.to_bytes(32, 'big')
    s = sig.s.to_bytes(32, 'big')
    for header in headers[ver]:
        signature = base64.b64encode(header + r + s).decode('utf-8')
        verified, _, _ = verify_message(address, message, signature)
        if verified:
            return address, message, signature
    raise SignatureError('Invalid signature parameters')


def bitcoin_message(address: str, message: str, signature: str, /) -> None:
    print('-----BEGIN BITCOIN SIGNED MESSAGE-----')
    print(f'{message}')
    print('-----BEGIN BITCOIN SIGNATURE-----')
    print(address)
    print()
    print(signature)
    print('-----END BITCOIN SIGNATURE-----')


def verify_message(address: str, message: str, signature: str, /) -> tuple[bool, str, str]:
    """ Verify signature with address and message """
    try:
        dsig = base64.b64decode(signature)
    except base64.binascii.Error as error:  # type: ignore
        raise SignatureError(f'Failed to decode signature: {error.args[0]}')
    if len(dsig) != 65:
        raise SignatureError('Signature must be 65 bytes long:', len(dsig))
    header, r, s = dsig[0], int.from_bytes(dsig[1:33], 'big'), int.from_bytes(dsig[33:], 'big')
    if header < 27 or header > 46:
        raise SignatureError('Header byte out of range:', header)
    if r >= secp256k1.n_curve or r == 0:
        raise SignatureError('r-value out of range:', r)
    if s >= secp256k1.n_curve or s == 0:
        raise SignatureError('s-value out of range:', s)
    uncompressed = False
    addr_type = 'p2pkh'
    if header >= 43:
        header -= 16
        addr_type = ''
    if header >= 39:
        header -= 12
        addr_type = 'p2wpkh'
    elif header >= 35:
        header -= 8
        addr_type = 'p2wpkh-p2sh'
    elif header >= 31:
        header -= 4
    else:
        uncompressed = True
    recid = header - 27
    x = r + secp256k1.n_curve * (recid >> 1)
    alpha = pow(x, 3) + secp256k1.b_curve % secp256k1.p_curve
    beta = pow(alpha, secp256k1.p_curve + 1 >> 2, secp256k1.p_curve)
    y = beta
    if is_odd(beta - recid):
        y = secp256k1.p_curve - beta
    R = Point(x, y)
    m_bytes = msg_magic(message)
    z = int.from_bytes(double_sha256(m_bytes), 'big')
    e = (-z) % secp256k1.n_curve
    inv_r = mod_inverse(r, secp256k1.n_curve)
    p = ec_mul(s, R)
    q = ec_mul(e, secp256k1.gen_point)
    Q = ec_add(p, q)
    raw_pubkey = to_affine(ec_mul(inv_r, Q))
    pubkey = create_pubkey(raw_pubkey, uncompressed=uncompressed)
    if addr_type:
        addr, _ = derive_address(pubkey, addr_type)
    else:
        raise SignatureError('Unknown address type')
    if addr == address:
        return True, pubkey.hex(), f'Message verified to be from {address}'
    return False, pubkey.hex(), 'Message failed to verify'


def main():
    parser = argparse.ArgumentParser(
        prog='python3 <application>',
        description='Bitcoin message signing/verification tool')
    subparsers = parser.add_subparsers()
    sign_parser = subparsers.add_parser('sign')
    sign_parser.set_defaults(cmd='sign')
    sign_group = sign_parser.add_argument_group(title='Sign messsage')
    sign_group.add_argument('-p',
                            '--privkey',
                            action=PrivateKey,
                            dest='privkey',
                            nargs=0,
                            required=True,
                            help='private key in wallet import format (WIF)')
    sign_group.add_argument('-a',
                            '--addr_type',
                            required=True,
                            choices=['p2pkh', 'p2wpkh-p2sh', 'p2wpkh'],
                            help='type of bitcoin address')
    sign_group.add_argument('-m',
                            '--message',
                            nargs='*',
                            required=True,
                            help='Message to sign')
    sign_group.add_argument('-d',
                            '--deterministic',
                            action='store_true',
                            help='sign deterministtically (RFC6979)')
    sign_group.add_argument('-v',
                            '--verbose',
                            action='store_true',
                            help='print prettified message')
    verify_parser = subparsers.add_parser('verify')
    verify_parser.set_defaults(cmd='verify')
    verify_group = verify_parser.add_argument_group(title='Verify messsage')
    verify_group.add_argument('-a',
                              '--address',
                              required=True,
                              help='specify bitcoin address')
    verify_group.add_argument('-m',
                              '--message',
                              nargs='*',
                              required=True,
                              help='Message to verify')
    verify_group.add_argument('-s',
                              '--signature',
                              required=True,
                              help='bitcoin signature in base64 format')
    verify_group.add_argument('-v',
                              '--verbose',
                              action='store_true',
                              help='print full message')
    verify_group.add_argument('-r',
                              '--recpub',
                              action='store_true',
                              help='recover public key')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)

    message = ' '.join(word for word in args.message)
    if args.cmd == 'sign':
        privkey = args.privkey
        result = sign_message(privkey, args.addr_type, message, deterministic=args.deterministic)
        if args.verbose:
            bitcoin_message(*result)
        else:
            print('Bitcoin address: {}\nMessage: {}\nSignature: {}'.format(*result))
    elif args.cmd == 'verify':
        verified, pubkey, result = verify_message(args.address, message, args.signature)
        print(verified)
        if args.verbose:
            print(result)
        if args.recpub:
            print(pubkey)


if __name__ == '__main__':
    main()

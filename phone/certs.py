"""A local certificate authority for the phone bridge, so Safari and Chrome treat the bridge's page as a secure
context (Geolocation, DeviceOrientation and getUserMedia need one). Pure Python (`cryptography`): the same on
Windows and macOS, no openssl or mkcert needed.

    ca = ensure_ca(state_dir)                       # once per laptop: install ca.crt on the phone, once
    cert, key = ensure_leaf(state_dir, ["192.168.137.1"], hostnames=["kite-pc"])   # every start; cheap

The CA is made once and kept (the phone trusts it after one profile install). The server certificate is re-issued
whenever this laptop's addresses change, signed by the same CA, so the phone never needs a second install.
Apple's rules for TLS server certificates (support.apple.com/103769): SAN required, EKU serverAuth, validity
<= 825 days, RSA >= 2048 or ECDSA. We use ECDSA P-256, SAN with every IP and name, EKU serverAuth, 397 days.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import os
import plistlib
import uuid
from dataclasses import dataclass
from pathlib import Path

CA_DAYS = 3650
LEAF_DAYS = 397
RENEW_DAYS = 30
CA_NAME = "Kite phone bridge local CA"


@dataclass
class CA:
    cert_path: Path
    key_path: Path
    created: bool

    @property
    def der(self) -> bytes:
        from cryptography import x509
        from cryptography.hazmat.primitives.serialization import Encoding
        return x509.load_pem_x509_certificate(self.cert_path.read_bytes()).public_bytes(Encoding.DER)


def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        os.chmod(path, 0o600)        # a no-op beyond read-only on Windows; the state folder is the user's own
    except OSError:
        pass


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def ensure_ca(state_dir: Path) -> CA:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    state_dir = Path(state_dir)
    cert_path, key_path = state_dir / "ca.crt", state_dir / "ca.key"
    if cert_path.is_file() and key_path.is_file():
        return CA(cert_path, key_path, False)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CA_NAME),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Kite (local, this laptop only)")])
    now = _now()
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=CA_DAYS))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    _write_private(key_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return CA(cert_path, key_path, True)


def _leaf_covers(cert, ips: list[str], names: list[str]) -> bool:
    from cryptography import x509
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return False
    have_ips = {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    have_names = {n.lower() for n in san.get_values_for_type(x509.DNSName)}
    left = cert.not_valid_after_utc - _now()
    return set(ips) <= have_ips and {n.lower() for n in names} <= have_names and left > dt.timedelta(days=RENEW_DAYS)


def ensure_leaf(state_dir: Path, ips: list[str], hostnames: list[str] = ()) -> tuple[Path, Path]:
    """The server certificate for these addresses and names (re-issued only when they change or it is near expiry)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    state_dir = Path(state_dir)
    ca = ensure_ca(state_dir)
    ips = sorted({*ips, "127.0.0.1"})
    names = sorted({*(h for h in hostnames if h), "localhost"})
    cert_path, key_path = state_dir / "server.crt", state_dir / "server.key"
    if cert_path.is_file() and key_path.is_file() and not ca.created:
        try:
            cur = x509.load_pem_x509_certificate(cert_path.read_bytes())
            if _leaf_covers(cur, ips, names):
                return cert_path, key_path
        except ValueError:
            pass
    ca_key = serialization.load_pem_private_key(ca.key_path.read_bytes(), password=None)
    ca_cert = x509.load_pem_x509_certificate(ca.cert_path.read_bytes())
    key = ec.generate_private_key(ec.SECP256R1())
    now = _now()
    san = [x509.IPAddress(ipaddress.ip_address(i)) for i in ips] + [x509.DNSName(n) for n in names]
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"Kite phone bridge {ips[0]}")]))
            .issuer_name(ca_cert.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=LEAF_DAYS))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False,
                                         content_commitment=False, key_encipherment=False, data_encipherment=False,
                                         key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    _write_private(key_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def mobileconfig(ca: CA) -> bytes:
    """An (unsigned) iOS configuration profile that installs the CA. iOS calls it "Unverified"; that is expected.
    After installing it: Settings > General > About > Certificate Trust Settings > turn on full trust for it."""
    ident = "dev.trykite.phone-bridge-ca"
    inner = {"PayloadType": "com.apple.security.root", "PayloadVersion": 1,
             "PayloadIdentifier": f"{ident}.cert", "PayloadUUID": str(uuid.uuid5(uuid.NAMESPACE_URL, ident + ".c")),
             "PayloadDisplayName": CA_NAME, "PayloadContent": ca.der}
    outer = {"PayloadType": "Configuration", "PayloadVersion": 1, "PayloadIdentifier": ident,
             "PayloadUUID": str(uuid.uuid5(uuid.NAMESPACE_URL, ident)), "PayloadDisplayName": "Kite phone bridge",
             "PayloadDescription": "Lets this phone open the Kite page served by your own laptop over HTTPS.",
             "PayloadOrganization": "Kite", "PayloadRemovalDisallowed": False, "PayloadContent": [inner]}
    return plistlib.dumps(outer, fmt=plistlib.FMT_XML)


def ssl_context(cert_path: Path, key_path: Path):
    import ssl
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx

use rand_core::OsRng;
use russh::keys::ssh_key::{HashAlg, PublicKey as SshPublicKey};
use russh::keys::{Algorithm, PrivateKey as SshPrivateKey};


/// get fingerprint of a public key
pub fn key_fingerprint_sha256(k: &SshPublicKey) -> String {
    k.fingerprint(HashAlg::Sha256).to_string()
}



/// mock testing fn for keypairs
pub fn generate_key_pair(
    _user: &String
) -> Result<(SshPublicKey, SshPrivateKey), anyhow::Error> {
    let private_key = SshPrivateKey::random(&mut OsRng, Algorithm::Ed25519)?;
    let public_key = private_key.public_key().clone();
    Ok((public_key, private_key))
}

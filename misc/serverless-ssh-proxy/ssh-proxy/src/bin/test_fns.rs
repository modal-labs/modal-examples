use log::info;
use ssh_proxy::key_handler::generate_key_pair;
use ssh_key::LineEnding;
use env_logger;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let (public_key, private_key) = generate_key_pair(&"aadit_juneja".to_string())?;
    let priv_key = private_key.to_openssh(LineEnding::LF)?;
    Ok(())
}

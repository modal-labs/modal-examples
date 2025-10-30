use std::{future::Future, sync::Arc, time::Duration};

use anyhow::bail;
use log::{info, warn};

use rand_core::OsRng;
use russh::Channel;
use russh::Preferred;
use russh::keys::ssh_key;
use russh::{
    ChannelId, MethodKind, MethodSet, Pty,
    keys::{Algorithm, Certificate, PrivateKey},
    server::{self, Auth, Config, Handler, Msg, Response, Server, Session},
};
use tokio::{sync::Mutex, task};


use ssh_proxy::key_handler::key_fingerprint_sha256;
use ssh_proxy::pty_connection_manager::PTYConnectionManager;
use ssh_proxy::database_manager::{create_dicts, add_entry, get_entry};
use ssh_proxy::server_ops::{ModalInternalCredentials, RpcClient, make_channel, AuthInterceptor};
use ssh_proxy::modal_proto::client::modal_client_client::ModalClientClient;




//  3 required environment variables: MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, SSH_PUBLIC_KEY. One optional environment variable: MODAL_ENVIRONMENT_NAME.

struct SSHServer {
    // a simpler data structure would be nice
    client: Option<Arc<Mutex<PTYConnectionManager>>>,
    user: Option<String>,
    dict_id: Option<String>,
}

impl SSHServer {
    async fn new() -> Result<Self, anyhow::Error> {
        // Create dict and seed ssh_public_key (from env) in the cloud dict
        let channel = make_channel().await?;
        let credentials = ModalInternalCredentials::new(None, None);
        let mut rpc_client: RpcClient = ModalClientClient::with_interceptor(
            channel,
            AuthInterceptor::new(credentials),
        );
        let dict_id = create_dicts(&mut rpc_client).await?;
        if let Ok(env_key) = std::env::var("SSH_PUBLIC_KEY") {
            let _ = add_entry(&mut rpc_client, &dict_id, "ssh_public_key", env_key.as_str()).await;
        }
        Ok(Self {
            client: None,
            user: None,
            dict_id: Some(dict_id),
        })
    }

    async fn post(&mut self, _channel: ChannelId, data: &[u8]) -> Result<(), anyhow::Error> {
        if let Some(pty_client) = self.client.clone() {
            let mut locked_client = pty_client.lock().await;
            locked_client.pty_command(data).await?;
            return Ok(());
        }
        bail!("No client found");
    }
}

impl russh::server::Server for SSHServer {
    type Handler = Self;

    fn new_client(&mut self, _peer_addr: Option<std::net::SocketAddr>) -> Self {
        let new_client = Self {
            client: None,
            user: None,
            dict_id: self.dict_id.clone(),
        };
        new_client
    }

    fn handle_session_error(&mut self, _error: <Self::Handler as russh::server::Handler>::Error) {
        warn!("Session error: {:?}", _error);
    }
}

impl Handler for SSHServer {
    type Error = anyhow::Error;

    async fn channel_open_session(
        &mut self,
        channel: Channel<Msg>,
        session: &mut Session,
    ) -> Result<bool, anyhow::Error> {
        self.client = Some(Arc::new(Mutex::new(
            PTYConnectionManager::new(
                Some(channel.id()),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .await,
        )));
        session.data(
            channel.id(),
            format!("Session opened for channel {}", channel.id())
                .as_bytes()
                .into(),
        )?;
        let cloned_conn = self.client.as_ref().unwrap().clone();
        let mut locked_conn = cloned_conn.lock().await;
        let init_handle = session.handle();
        if self.user.is_none() {
            bail!("User not provided");
        }

        // Read image_id from cloud dict (if present)
        let mut image_id_opt: Option<String> = None;
        if let Some(dict_id) = &self.dict_id {
            let channel = make_channel().await?;
            let credentials = ModalInternalCredentials::new(None, None);
            let mut rpc_client: RpcClient = ModalClientClient::with_interceptor(
                channel,
                AuthInterceptor::new(credentials),
            );
            image_id_opt = get_entry(&mut rpc_client, dict_id.as_str(), "image_id").await?;
            info!(" we have the image_id_opt: {:?}", image_id_opt);
        }

        let res = locked_conn
            .connection_prologue(init_handle, image_id_opt)
            .await;

        if let Err(e) = res.as_ref() {
            warn!("Error starting connection prologue: {}", e);
        }
        let stdout_rx = res.unwrap();
        let channel_id = channel.id();
        let session_handle = session.handle();
        task::spawn(async move {
            loop {
                match stdout_rx.recv_async().await {
                    Ok(batch) => {
                        let message_bytes = batch
                            .clone()
                            .items
                            .iter()
                            .map(|item| item.message_bytes.clone())
                            .collect::<Vec<_>>();
                        for chunk in message_bytes {
                            let send_res =
                                session_handle.data(channel_id, chunk.into()).await;
                            match send_res {
                                Err(e) => {
                                    warn!("Error sending data: {:?}", e);
                                    break;
                                }
                                Ok(_) => {
                                    continue;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Error getting stdout: {}", e);
                        break;
                    }
                }
            }
        });
        Ok(true)
    }

    async fn data(
        &mut self,
        channel: ChannelId,
        data: &[u8],
        _session: &mut Session,
    ) -> Result<(), anyhow::Error> {
        if data == [3] {
            // ctrl+c
            self.channel_close(channel, _session).await?;
        } else {
            self.post(channel, data).await?;
        }
        Ok(())
    }

    fn channel_close(
        &mut self,
        channel: ChannelId,
        _session: &mut Session,
    ) -> impl Future<Output = Result<(), anyhow::Error>> {
        info!("channel close received for channel {}. Bye!", channel);
        let cloned_client = self.client.as_ref().unwrap().clone();
        let dict_id_opt = self.dict_id.clone();
        async move {
            let image_id = cloned_client.lock().await.connection_epilogue().await?;
            info!("FS snapshot image id: {} on channel {}", image_id, channel);
            // Persist image_id in cloud dict
            if let Some(dict_id) = dict_id_opt.as_ref() {
                let channel = make_channel().await?;
                let credentials = ModalInternalCredentials::new(None, None);
                let mut rpc_client: RpcClient = ModalClientClient::with_interceptor(
                    channel,
                    AuthInterceptor::new(credentials),
                );
                let _ = add_entry(&mut rpc_client, dict_id.as_str(), "image_id", image_id.as_str())
                    .await;
            }
            Ok(())
        }
    }

    #[allow(unused_variables)]
    fn auth_none(&mut self, _user: &str) -> impl Future<Output = Result<Auth, Self::Error>> + Send {
        async {
            info!("auth_none received for user: {}", _user.to_string());
            self.user = Some(_user.to_string());
            info!("got user: {}", _user.to_string());
            Ok(Auth::Reject {
                partial_success: false,
                proceed_with_methods: Some(MethodSet::from(vec![MethodKind::PublicKey].as_slice())),
            })
        }
    }

    #[allow(unused_variables)]
    fn auth_password(
        &mut self,
        user: &str,
        password: &str,
    ) -> impl Future<Output = Result<Auth, Self::Error>> + Send {
        async {
            return Ok(Auth::Reject {
                partial_success: false,
                proceed_with_methods: Some(MethodSet::from(vec![MethodKind::PublicKey].as_slice())),
            });
        }
    }

    #[allow(unused_variables)]
    fn auth_publickey_offered(
        &mut self,
        user: &str,
        public_key: &ssh_key::PublicKey,
    ) -> impl Future<Output = Result<Auth, Self::Error>> + Send {
        let dict_id_opt = self.dict_id.clone();
        info!("offered_key: {:?}", public_key.to_openssh().unwrap_or_default());
        let offered_fp = key_fingerprint_sha256(public_key);
        async move {
            // Read allowed ssh_public_key from cloud dict and validate
            if let Some(dict_id) = dict_id_opt.as_ref() {
                let channel = make_channel().await?;
                let credentials = ModalInternalCredentials::new(None, None);
                let mut rpc_client: RpcClient = ModalClientClient::with_interceptor(
                    channel,
                    AuthInterceptor::new(credentials),
                );
                if let Some(stored_key) = get_entry(&mut rpc_client, dict_id.as_str(), "ssh_public_key").await? {
                    info!("stored_key: {:?}", stored_key);
                    match ssh_key::PublicKey::from_openssh(stored_key.as_str()) {
                        Ok(parsed) => {
                            if key_fingerprint_sha256(&parsed) == offered_fp {
                                info!("accepted public key");
                                return Ok(Auth::Accept);
                            } else {
                                warn!("offered key does not match stored key");
                                return Ok(Auth::Reject {
                                    partial_success: false,
                                    proceed_with_methods: Some(MethodSet::from(vec![MethodKind::PublicKey].as_slice())),
                                });
                            }
                        }
                        Err(_) => {
                            warn!("error when trying to get stored key");
                            return Ok(Auth::Reject {
                                partial_success: false,
                                proceed_with_methods: Some(MethodSet::from(vec![MethodKind::PublicKey].as_slice())),
                            });
                        }
                    }
                }
            }
            info!("no stored key found");
            info!("rejected public key");
            Ok(Auth::Reject {
                partial_success: false,
                proceed_with_methods: None,
            })
        }
    }

    #[allow(unused_variables)]
    fn auth_keyboard_interactive<'a>(
        &'a mut self,
        user: &str,
        submethods: &str,
        response: Option<Response<'a>>,
    ) -> impl Future<Output = Result<Auth, Self::Error>> + Send {
        async {
            Ok(Auth::Reject {
                partial_success: false,
                proceed_with_methods: None,
            })
        }
    }

    #[allow(unused_variables)]
    async fn auth_publickey(
        &mut self,
        _user: &str,
        key: &ssh_key::PublicKey,
    ) -> Result<server::Auth, Self::Error> {
        self.user = Some(_user.to_string());
        info!("auth_publickey received for user: {}", _user.to_string());
        
        if let Some(dict_id) = self.dict_id.as_ref() {
            let channel = make_channel().await?;
            let credentials = ModalInternalCredentials::new(None, None);
            let mut rpc_client: RpcClient = ModalClientClient::with_interceptor(
                channel,
                AuthInterceptor::new(credentials),
            );
            if let Some(stored_key) = get_entry(&mut rpc_client, dict_id.as_str(), "ssh_public_key").await? {
                info!("stored_key: {:?}", stored_key);
                let offered_fp = key_fingerprint_sha256(key);
                match ssh_key::PublicKey::from_openssh(stored_key.as_str()) {
                    Ok(parsed) => {
                        if key_fingerprint_sha256(&parsed) == offered_fp {
                            info!("accepted public key");
                            return Ok(Auth::Accept);
                        } else {
                            warn!("offered key does not match stored key");
                            return Ok(Auth::Reject {
                                partial_success: false,
                                proceed_with_methods: Some(MethodSet::from(vec![MethodKind::PublicKey].as_slice())),
                            });
                        }
                    }
                    Err(_) => {
                        warn!("error when trying to get stored key");
                        return Ok(Auth::Reject {
                            partial_success: false,
                            proceed_with_methods: Some(MethodSet::from(vec![MethodKind::PublicKey].as_slice())),
                        });
                    }
                }
            } else {
                return Ok(Auth::Reject {
                    partial_success: false,
                    proceed_with_methods: None,
                });
            }
        } else {
            return Ok(Auth::Reject {
                partial_success: false,
                proceed_with_methods: None,
            });
        }
    }

    async fn auth_openssh_certificate(
        &mut self,
        _user: &str,
        _certificate: &Certificate,
    ) -> Result<server::Auth, Self::Error> {
        info!(
            "auth_openssh_certificate received for user: {}",
            _user.to_string()
        );
        self.user = Some(_user.to_string());
        Ok(server::Auth::Reject {
            partial_success: false,
            proceed_with_methods: None,
        })
    }

    #[allow(unused_variables, clippy::too_many_arguments)]
    fn pty_request(
        &mut self,
        channel: ChannelId,
        term: &str,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
        modes: &[(Pty, u32)],
        session: &mut Session,
    ) -> impl Future<Output = Result<(), anyhow::Error>> {
        async {
            Ok(())
        }
    }

    #[allow(unused_variables)]
    fn shell_request(
        &mut self,
        channel: ChannelId,
        session: &mut Session,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        async {
            Ok(())
        }
    }

    #[allow(unused_variables)]
    fn exec_request(
        &mut self,
        channel: ChannelId,
        data: &[u8],
        session: &mut Session,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        async move {
            let conn = self.client.as_ref().unwrap().clone();
            let mut locked_conn = conn.lock().await;
            locked_conn.pty_command(data).await?;
            Ok(())
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut server = SSHServer::new().await?;
    let config = Config {
        inactivity_timeout: Some(Duration::from_secs(3600)),
        auth_rejection_time: Duration::from_secs(3),
        auth_rejection_time_initial: Some(Duration::from_secs(0)),
        keys: vec![PrivateKey::random(&mut OsRng, Algorithm::Ed25519).unwrap()],
        preferred: Preferred {
            ..Preferred::default()
        },
        ..Default::default()
    };
    info!("Running server on 0.0.0.0:22");
    server
        .run_on_address(Arc::new(config), "0.0.0.0:22")
        .await
        .unwrap();
    Ok(())
}

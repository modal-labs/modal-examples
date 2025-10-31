use std::str::FromStr;

use crate::modal_proto::client::modal_client_client::ModalClientClient;
use tonic::{
    Request, Status,
    metadata::MetadataValue,
    service::{Interceptor, interceptor::InterceptedService},
    transport::{Channel, Endpoint},
};

/// Minimally required credentials to authenticate with the modal client api
/// server.
pub struct ModalInternalCredentials {
    token_id: String,
    token_secret: String,
    client_version: String,
}

/// Implements the constructor for the modal internal credentials.
impl ModalInternalCredentials {
    /// Implements the constructor for the modal internal credentials.
    pub fn new(token_id: Option<String>, token_secret: Option<String>) -> Self {
        Self {
            token_id: token_id.unwrap_or_else(|| std::env::var("TOKEN_ID").unwrap()),
            token_secret: token_secret
                .unwrap_or_else(|| std::env::var("TOKEN_SECRET").unwrap()),
            client_version: "1.2.0".to_string(),
        }
    }
}

/// Injects auth headers into the request
pub fn inject_auth_headers(
    credentials: &ModalInternalCredentials,
) -> impl tonic::service::Interceptor + use<'_> {
    |mut request: tonic::Request<()>| {
        let metadata = request.metadata_mut();
        metadata.insert(
            "x-modal-token-id",
            MetadataValue::from_str(credentials.token_id.as_str()).unwrap(),
        );
        metadata.insert(
            "x-modal-token-secret",
            MetadataValue::from_str(credentials.token_secret.as_str()).unwrap(),
        );
        metadata.insert(
            "x-modal-client-version",
            MetadataValue::from_str(credentials.client_version.as_str()).unwrap(),
        );
        Ok(request)
    }
}

/// Makes a channel to the modal client api server.
pub async fn make_channel() -> Result<Channel, anyhow::Error> {
    let external_api_endpoint = String::from("http://api.modal.com");
    Ok(Endpoint::from_shared(external_api_endpoint)?
        .connect()
        .await?)
}

/// Wrapper type for the auth interceptor. Needed to get a type definition for
/// gRPC client which is used in terminal_handler.rs
pub struct AuthInterceptor {
    credentials: ModalInternalCredentials,
}

impl AuthInterceptor {
    /// Create new auth interceptor ( easier to reuse credentials across gRPC
    /// requests)
    pub fn new(credentials: ModalInternalCredentials) -> Self {
        Self { credentials }
    }
}

impl Interceptor for AuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        let metadata = request.metadata_mut();
        metadata.insert(
            "x-modal-token-id",
            MetadataValue::from_str(self.credentials.token_id.as_str()).unwrap(),
        );
        metadata.insert(
            "x-modal-token-secret",
            MetadataValue::from_str(self.credentials.token_secret.as_str()).unwrap(),
        );
        metadata.insert(
            "x-modal-client-version",
            MetadataValue::from_str(self.credentials.client_version.as_str()).unwrap(),
        );
        Ok(request)
    }
}

/// gRPC client type wrapper
pub type RpcClient = ModalClientClient<InterceptedService<Channel, AuthInterceptor>>;

use tonic::Request;

use crate::server_ops::RpcClient;

use crate::modal_proto::client::{
    DictEntry, DictGetOrCreateRequest, DictGetRequest, DictUpdateRequest,
};
use uuid::Uuid;

use log::info; 


/// Create (or fetch) the single dict used to store session metadata.
pub async fn create_dicts(rpc_client: &mut RpcClient) -> Result<String, anyhow::Error> {
    let dict_id = "proxy-metadata-".to_string() + Uuid::new_v4().to_string().as_str();
    info!("Creating dict with id: {}", dict_id);
    let request = Request::new(DictGetOrCreateRequest {
        deployment_name: dict_id.clone(),
        environment_name: std::env::var("MODAL_ENVIRONMENT_NAME").unwrap_or("main".to_string()),
        object_creation_type: 1,
        data: vec![],
    });
    let response = rpc_client.dict_get_or_create(request).await?;
    Ok(response.into_inner().dict_id)
}

/// Upsert a metadata entry into the dict (key -> value)
pub async fn add_entry(
    rpc_client: &mut RpcClient,
    dict_id: &str,
    key: &str,
    value: &str,
) -> Result<(), anyhow::Error> {
    let request = Request::new(DictUpdateRequest {
        dict_id: dict_id.to_string(),
        if_not_exists: false,
        updates: vec![DictEntry {
            key: key.as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
        }],
    });
    let _ = rpc_client.dict_update(request).await?;
    Ok(())
}

/// Fetch a single entry by key from the dict
pub async fn get_entry(
    rpc_client: &mut RpcClient,
    dict_id: &str,
    key: &str,
) -> Result<Option<String>, anyhow::Error> {
    let request = Request::new(DictGetRequest {
        dict_id: dict_id.to_string(),
        key: key.as_bytes().to_vec(),
    });
    let response = rpc_client.dict_get(request).await?;
    let value = response.into_inner().value.map(|v| String::from_utf8(v).unwrap());
    info!("Got entry: {:?}", value);
    Ok(value)
}
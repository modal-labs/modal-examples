#![deny(clippy::all, clippy::let_underscore_future, clippy::unused_result_ok)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod pty_connection_manager;
pub mod server_ops;
pub mod key_handler;
pub mod database_manager;

include!(concat!(env!("OUT_DIR"), "/modal.client.rs"));



pub mod modal_proto {
    pub use crate::modal_client_client;
    pub mod client {
        pub use crate::*;
        pub use crate::modal_client_client::ModalClientClient;
    }
}

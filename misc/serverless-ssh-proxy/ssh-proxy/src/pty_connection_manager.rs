use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

use anyhow::bail;
use flume;
use log::{info, warn};
use russh::ChannelId;
use russh::server::Handle;
use tokio::sync::Mutex;
use tokio::time::Duration;
use tokio::{task, time::sleep};
use tonic::Request;

use crate::modal_proto::client::modal_client_client::ModalClientClient;
use crate::modal_proto::client::{
    AppCreateRequest, ContainerExecGetOutputRequest, ContainerExecPutInputRequest,
    ContainerExecRequest, Image, ImageContextFile, ImageGetOrCreateRequest,
    ImageJoinStreamingRequest, ImageRegistryConfig, PtyInfo, RuntimeInputMessage,
    RuntimeOutputBatch, Sandbox, SandboxCreateRequest, SandboxGetTaskIdRequest, SandboxSnapshotFsRequest, SandboxTerminateRequest,
};
use crate::server_ops;

/// client vers for image builder
pub const CLIENT_VERSION: &str = "2025.06";

/// Manages a PTY connection for a single user session
pub struct PTYConnectionManager {
    _channel_id: Option<ChannelId>,
    sandbox_id: String,
    pty_exec_id: String, // exec_id for the PTY proess
    _token_id: Option<String>,
    _token_secret: Option<String>,
    app_id: String,
    environment_name: String,
    rpc_client: Arc<Mutex<server_ops::RpcClient>>,
    message_index: u64,
    is_active: bool,
    timeout: f32,
    task_id: String,
    image_id: String,
    /// pty config
    pub term: String,
    /// pty config
    pub colorterm: String,
    /// pty config
    pub term_program: String,
    /// pty config
    pub rows: u32,
    /// pty config
    pub cols: u32,
}

impl PTYConnectionManager {
    /// Creates a new PTYConnectionManager
    pub async fn new(
        _channel_id: Option<ChannelId>,
        token_id: Option<String>,
        token_secret: Option<String>,
        sandbox_id: Option<String>,
        app_id: Option<String>,
        timeout: Option<f32>,
        task_id: Option<String>,
        image_id: Option<String>,
        term: Option<String>,
        colorterm: Option<String>,
        term_program: Option<String>,
        rows: Option<u32>,
        cols: Option<u32>,
    ) -> Self {
        let credentials =
            server_ops::ModalInternalCredentials::new(token_id.clone(), token_secret.clone());
        let channel = server_ops::make_channel().await.unwrap();
        Self {
            _channel_id: _channel_id,
            message_index: 1,
            sandbox_id: sandbox_id.unwrap_or_default().to_string(),
            _token_id: token_id.clone(),
            _token_secret: token_secret.clone(),
            app_id: app_id.clone().unwrap_or_default(),
            environment_name: std::env::var("MODAL_ENVIRONMENT_NAME").unwrap_or("main".to_string()),
            pty_exec_id: "".to_string(),
            rpc_client: Arc::new(Mutex::new(ModalClientClient::with_interceptor(
                channel,
                server_ops::AuthInterceptor::new(credentials),
            ))),
            is_active: true,
            timeout: timeout.unwrap_or(60.0 * 30.0), // 30 minutes
            task_id: task_id.clone().unwrap_or_default(),
            image_id: image_id.clone().unwrap_or_default(),
            term: term.clone().unwrap_or("xterm-256color".to_string()),
            colorterm: colorterm.clone().unwrap_or("truecolor".to_string()),
            term_program: term_program.clone().unwrap_or("vscode".to_string()),
            rows: rows.clone().unwrap_or(26),
            cols: cols.clone().unwrap_or(228),
        }
    }

    /// Spawns a task to read stdout from the PTY process
    pub async fn connection_prologue(
        &mut self,
        message_handle: Handle,
        image_id: Option<String>,
    ) -> Result<flume::Receiver<RuntimeOutputBatch>, anyhow::Error> {
        let _ = self.make_app().await;
        if let Some(id) = image_id {
            let _ = message_handle
                .data(
                    self._channel_id.unwrap(),
                    "Restoring from snapshot...\n".as_bytes().into(),
                )
                .await;
            self.image_id = id.clone();
        } else {
            let _ = self.make_image(message_handle.clone()).await;
        }
        let _ = self.create_sandbox().await;
        let _ = self.open_pty_process().await;
        // make the tx, rx inside this due to ownership issues
        let (stdout_tx, stdout_rx) = flume::bounded(1024);

        // Here we need to make a new client b/c stdout reader lives for the
        // lifetime of the SSH session, and it requires an exclusive
        // lock on a stream. Maybe there is a way to do this such that the
        // client is reused and ONLY the stream is held via an
        // exclusive lock? Will revisit this later
        let mut client = ModalClientClient::with_interceptor(
            server_ops::make_channel().await.unwrap(),
            server_ops::AuthInterceptor::new(server_ops::ModalInternalCredentials::new(None, None)),
        );
        let pty_exec_id = self.pty_exec_id.clone();
        let timeout = self.timeout;
        task::spawn(async move {
            let mut last_batch_index = 0;
            loop {
                let stdout_stream_req = Request::new(ContainerExecGetOutputRequest {
                    exec_id: pty_exec_id.clone(),
                    timeout: timeout,
                    last_batch_index: last_batch_index,
                    file_descriptor: 1,
                    get_raw_bytes: true,
                });
                let mut message_seen: bool = false;
                match client.container_exec_get_output(stdout_stream_req).await {
                    Ok(stdout_rx_res) => {
                        let mut stream = stdout_rx_res.into_inner();
                        loop {
                            match stream.message().await {
                                Ok(Some(batch)) => {
                                    last_batch_index = batch.batch_index;
                                    message_seen = true;
                                    stdout_tx.send(batch).unwrap();
                                }
                                Ok(None) => {
                                    break;
                                }
                                Err(e) => {
                                    warn!("Error getting stdout: {}", e);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Error getting stdout: {}", e);
                        break;
                    }
                }
                if !message_seen {
                    sleep(Duration::from_secs(1)).await; //avoid resource wastage
                }
            }
        });
        Ok(stdout_rx)
    }

    /// Performs an FS snapshot of the sandbox, kills it, and returns the image
    /// ID.
    pub async fn connection_epilogue(&mut self) -> Result<String, anyhow::Error> {
        let snapshot_req = Request::new(SandboxSnapshotFsRequest {
            sandbox_id: self.sandbox_id.clone(),
            timeout: 60.0,
        });
        let response = self
            .rpc_client
            .clone()
            .lock()
            .await
            .sandbox_snapshot_fs(snapshot_req)
            .await?;
        // kill the sandbox
        info!("snapshhotted now killing sandbox");
        let kill_req = Request::new(SandboxTerminateRequest {
            sandbox_id: self.sandbox_id.clone(),
        });
        let _kill_response = self
            .rpc_client
            .clone()
            .lock()
            .await
            .sandbox_terminate(kill_req)
            .await?;
        Ok(response.into_inner().image_id)
    }

    /// Creates sandbox, assigns id to instance variable
    pub async fn create_sandbox(&mut self) -> Result<String, anyhow::Error> {
        info!("Creating sandbox with app_id: {}", self.app_id);
        let req = Request::new(SandboxCreateRequest {
            app_id: self.app_id.clone(),
            environment_name: self.environment_name.clone(),
            // TODO(aadit-juneja): add more robust config support
            definition: Some(Sandbox {
                image_id: self.image_id.clone(),
                timeout_secs: self.timeout as u32,
                enable_snapshot: true,
                entrypoint_args: vec!["/bin/sh".to_string()],
                ..Default::default()
            }),
        });
        let response = self
            .rpc_client
            .clone()
            .lock()
            .await
            .sandbox_create(req)
            .await?;
        self.sandbox_id = response.into_inner().sandbox_id;
        // need to get sandbox task_id now. this is different from the sandbox_id.
        self.get_task_id().await?;
        Ok(self.sandbox_id.clone())
    }

    async fn get_task_id(&mut self) -> Result<String, anyhow::Error> {
        if self.sandbox_id.is_empty() {
            bail!("Sandbox ID is empty; can't get task ID.");
        }
        let req = Request::new(SandboxGetTaskIdRequest {
            sandbox_id: self.sandbox_id.clone(),
            timeout: Some(60.0),
            wait_until_ready: true,
        });
        let task_id_res = self
            .rpc_client
            .clone()
            .lock()
            .await
            .sandbox_get_task_id(req)
            .await?;
        let task_id = task_id_res.into_inner().task_id.unwrap_or_default();
        self.task_id = task_id.clone();
        Ok(task_id)
    }

    /// execute a command in non-PTY mode
    pub async fn exec_command(&mut self, command: &String) -> Result<String, anyhow::Error> {
        if !self.is_active {
            bail!("Sandbox is inactive; can't execute command.");
        }
        let req = Request::new(ContainerExecRequest {
            task_id: self.task_id.clone(),
            command: vec![command.clone()],
            pty_info: None,
            stdout_output: 2,
            stderr_output: 2,
            timeout_secs: self.timeout as u32,
            ..Default::default() /* terminate_container_on_exit field is deprecated but still
                                  * required? seemed most ergonomic way to handle this */
        });
        let response = self
            .rpc_client
            .clone()
            .lock()
            .await
            .container_exec(req)
            .await?;
        let exec_id = response.into_inner().exec_id;
        Ok(exec_id)
    }

    /// Opens a PTY process in the sandbox.
    pub async fn open_pty_process(&mut self) -> Result<String, anyhow::Error> {
        if !self.is_active {
            bail!("Sandbox is inactive; can't open PTY process.");
        }

        let req = Request::new(ContainerExecRequest {
            task_id: self.task_id.clone(),
            command: vec!["/bin/bash".to_string()],
            pty_info: Some(PtyInfo {
                winsz_rows: self.rows,
                winsz_cols: self.cols,
                env_term: self.term.clone(),
                env_colorterm: self.colorterm.clone(),
                env_term_program: self.term_program.clone(),
                pty_type: 2, // shell
                ..Default::default()
            }),
            stdout_output: 2,
            stderr_output: 2,
            timeout_secs: self.timeout as u32,
            ..Default::default()
        });
        let response = self
            .rpc_client
            .clone()
            .lock()
            .await
            .container_exec(req)
            .await?;
        let exec_id = response.into_inner().exec_id;
        self.pty_exec_id = exec_id.clone();
        Ok(exec_id)
    }

    /// Sends a command to the PTY process
    pub async fn pty_command(&mut self, command: &[u8]) -> Result<(), anyhow::Error> {
        if !self.is_active {
            bail!("Sandbox is inactive; can't send command.");
        }
        let req = Request::new(ContainerExecPutInputRequest {
            exec_id: self.pty_exec_id.clone(),
            input: Some(RuntimeInputMessage {
                message: command.to_vec().into(),
                eof: false,
                message_index: self.message_index,
            }),
        });

        self.rpc_client
            .clone()
            .lock()
            .await
            .container_exec_put_input(req)
            .await?;
        self.message_index += 1; // satisfy increasing index requirement
        Ok(())
    }

    /// Makes a default app
    pub async fn make_app(&mut self) -> Result<String, anyhow::Error> {
        let app_req = Request::new(AppCreateRequest {
            app_state: 1,
            environment_name: self.environment_name.clone(),
            description: "myssh".to_string(),
            ..Default::default()
        });
        let app_response = self
            .rpc_client
            .clone()
            .lock()
            .await
            .app_create(app_req)
            .await?;
        let app_id = app_response.into_inner().app_id;
        self.app_id = app_id.clone();
        Ok(app_id)
    }

    /// Makes a default debian slim image
    pub async fn make_image(&mut self, message_handle: Handle) -> Result<String, anyhow::Error> {
        // builder directory is the same as it is in the client repo
        let mut abs_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        abs_path.push("builder");
        abs_path.push(format!("{}.txt", CLIENT_VERSION));
        let requirements_data = fs::read(&abs_path)?;
        let full_python = "3.11.10".to_string();
        let debian_codename = "bookworm".to_string();
        let dockerfile_commands: Vec<String> = vec![
            format!("FROM python:{full_python}-slim-{debian_codename}").to_string(),
            format!("COPY /modal_requirements.txt /modal_requirements.txt").to_string(),
            "RUN apt-get update".to_string(),
            "RUN apt-get install -y gcc gfortran build-essential".to_string(),
            "RUN pip install --upgrade pip wheel uv".to_string(),
            "RUN uv pip install --system --compile-bytecode --no-cache --no-deps -r \
             /modal_requirements.txt"
                .to_string(),
            "RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections"
                .to_string(),
            "RUN rm /modal_requirements.txt".to_string(),
            r#"CMD ["sleep", "infinity"]"#.to_string(),
        ];
        let image_def = Image {
            base_images: vec![],
            dockerfile_commands: dockerfile_commands,
            context_files: vec![ImageContextFile {
                filename: "/modal_requirements.txt".to_string(),
                data: requirements_data.into(),
            }],
            secret_ids: vec![],
            context_mount_id: "".to_string(),
            gpu_config: None,
            image_registry_config: Some(ImageRegistryConfig {
                secret_id: "".to_string(),
                registry_auth_type: 0, // RegistryAuthType::Unspecified (the default)
            }),
            runtime: "gvisor".to_string(),
            runtime_debug: false,
            build_function: None,
            build_args: HashMap::new(),
            ..Default::default()
        };
        let req: Request<ImageGetOrCreateRequest> = Request::new(ImageGetOrCreateRequest {
            app_id: self.app_id.clone(),
            image: Some(image_def),
            existing_image_id: "".to_string(),
            build_function_id: "".to_string(),
            force_build: false,
            builder_version: CLIENT_VERSION.to_string(),
            allow_global_deployment: false,
            ignore_cache: false,
            namespace: 3,
        });
        let response = self
            .rpc_client
            .clone()
            .lock()
            .await
            .image_get_or_create(req)
            .await?;
        let image_id = response.into_inner().image_id.clone();
        self.image_id = image_id.clone();
        // wait for image to be built before returning
        let req = Request::new(ImageJoinStreamingRequest {
            image_id: image_id.clone(),
            timeout: 600.0,
            ..Default::default()
        });
        let mut msg_stream = self
            .rpc_client
            .clone()
            .lock()
            .await
            .image_join_streaming(req)
            .await?
            .into_inner();
        loop {
            match msg_stream.message().await {
                Ok(Some(msg)) => {
                    if msg.eof {
                        break;
                    }
                    let _ = message_handle
                        .data(
                            self._channel_id.unwrap(),
                            msg.task_logs
                                .iter()
                                .map(|log| log.data.clone())
                                .collect::<Vec<_>>()
                                .join("\n")
                                .as_bytes()
                                .into(),
                        )
                        .await;
                }
                Ok(None) => {
                    break;
                }
                Err(e) => {
                    warn!("Error getting image join streaming response: {}", e);
                    break;
                }
            }
        }
        Ok(image_id)
    }
}

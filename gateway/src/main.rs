use config::Config;
use langdb_core::error::GatewayError;
use rest::ApiServer;
use thiserror::Error;

mod config;
mod cost;
mod otel;
mod rest;
mod tracing;
#[derive(Error, Debug)]
pub enum CliError {
    #[error(transparent)]
    GatewayError(#[from] GatewayError),
    #[error(transparent)]
    ConfigError(#[from] Box<dyn std::error::Error>),
}

#[actix_web::main]
async fn main() -> Result<(), CliError> {
    tracing::init_tracing();

    std::env::set_var("RUST_BACKTRACE", "1");

    let config = Config::load("config.yaml");

    let api_server = ApiServer::new(config);

    let api_result = api_server.start().await.unwrap().await;
    if let Err(e) = api_result {
        eprintln!("API Server Error: {:?}", e);
    }
    Ok(())
}

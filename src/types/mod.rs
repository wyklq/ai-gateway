pub mod aws;
pub mod credentials;
pub mod embed;
pub mod engine;
pub mod gateway;
pub mod image;
pub mod json;
pub mod message;
pub mod provider;
pub mod threads;

pub type GatewayResult<T> = Result<T, crate::error::GatewayError>;

#[derive(Clone, Debug)]
pub struct GatewayTenant {
    pub name: String,
    pub project_slug: String,
}

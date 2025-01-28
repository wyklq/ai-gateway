use langdb_core::models::LlmModelDefinition;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RestConfig {
    pub host: String,
    pub port: u16,
    pub cors_allowed_origins: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub rest: RestConfig,
    pub models: Vec<LlmModelDefinition>,
}

impl Default for RestConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_allowed_origins: vec!["*".to_string()],
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rest: RestConfig::default(),
            models: Vec::new(),
        }
    }
}

impl Config {
    pub fn load<P: AsRef<Path>>(config_path: P) -> Self {
        match std::fs::File::open(config_path) {
            Ok(f) => {
                let a = serde_yaml::from_reader(f).unwrap();
                a
            }
            Err(_) => Self::default(),
        }
    }
}

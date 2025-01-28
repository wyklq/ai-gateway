use std::collections::HashMap;
use std::fmt::Display;
use std::str::FromStr;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_with::serde_as;
use serde_with::DisplayFromStr;

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum Credentials {
    ApiKey(#[serde_as(as = "DisplayFromStr")] ApiKeyCredentials),
    Aws(AwsCredentials),
    // Hosted LangDB AWS
    #[serde(other)]
    LangDb,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct IntegrationCredentials {
    pub secrets: HashMap<String, Value>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyCredentials {
    pub api_key: String,
}

impl Display for ApiKeyCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.api_key)
    }
}

impl FromStr for ApiKeyCredentials {
    type Err = std::string::ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ApiKeyCredentials {
            api_key: s.to_string(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AwsCredentials {
    pub access_key: String,
    pub access_secret: String,
    // Defaults tp us-east-1
    pub region: Option<String>,
}

use std::collections::HashMap;

use crate::types::gateway::ChatCompletionTool;

use crate::types::gateway::FunctionParameters;

pub struct GatewayTool {
    pub def: ChatCompletionTool,
}

#[async_trait::async_trait]
pub trait Tool: Send + Sync + 'static {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn get_function_parameters(&self) -> Option<FunctionParameters>;
    async fn run(
        &self,
        input: HashMap<String, serde_json::Value>,
        tags: HashMap<String, String>,
    ) -> crate::types::GatewayResult<serde_json::Value>;
    fn stop_at_call(&self) -> bool {
        false
    }
}

#[async_trait::async_trait]
impl Tool for GatewayTool {
    fn name(&self) -> String {
        self.def.function.name.to_string()
    }

    fn description(&self) -> String {
        self.def
            .function
            .description
            .clone()
            .unwrap_or("".to_string())
    }

    fn get_function_parameters(&self) -> std::option::Option<FunctionParameters> {
        Some(self.def.function.parameters.clone())
    }

    async fn run(
        &self,
        _inputs: HashMap<String, serde_json::Value>,
        _tags: HashMap<String, String>,
    ) -> crate::types::GatewayResult<serde_json::Value> {
        panic!("Gateway tool should not be called directly");
    }

    fn stop_at_call(&self) -> bool {
        true
    }
}

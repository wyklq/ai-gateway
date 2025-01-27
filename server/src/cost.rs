use langdb_ai_gateway::types::gateway::{
    CostCalculationResult, CostCalculator, CostCalculatorError, Usage,
};

pub struct DummyCostCalculator {}

#[async_trait::async_trait]
impl CostCalculator for DummyCostCalculator {
    async fn calculate_cost(
        &self,
        _model_name: &str,
        _provider_name: &str,
        _usage: &Usage,
    ) -> Result<CostCalculationResult, CostCalculatorError> {
        Ok(CostCalculationResult {
            cost: 0.0,
            per_input_token: 0.0,
            per_output_token: 0.0,
            per_image_cost: None,
        })
    }
}

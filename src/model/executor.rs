use serde_json::Value;
use std::{borrow::Cow, collections::HashMap};

use crate::{
    error::GatewayError,
    types::{
        engine::{InputArgs, ParentCompletionOptions},
        GatewayResult,
    },
};

pub fn prepare_input_variables(
    model_options: ParentCompletionOptions,
    inputs: Vec<Value>,
) -> GatewayResult<HashMap<String, Value>> {
    let definition = model_options.definition;
    let input_args = definition.get_input_args();

    let required_variables = definition.get_variables();

    let mut inputs_map = HashMap::new();
    for (ma, v) in input_args.iter().zip(inputs) {
        inputs_map.insert(ma.name.clone(), v);
    }

    validate_variables(&inputs_map, required_variables, &input_args)?;

    tracing::debug!("Input Variables: {:?}", inputs_map.clone());

    Ok(inputs_map)
}

pub fn validate_variables(
    input_variables: &HashMap<String, Value>,
    required_variables: Vec<Cow<'_, String>>,
    model_args: &InputArgs,
) -> GatewayResult<()> {
    for var in required_variables {
        let var = var.to_string();
        if !model_args.contains(&var) && !input_variables.contains_key(&var) {
            return Err(GatewayError::MissingVariable(var.clone()));
        }
    }
    Ok(())
}

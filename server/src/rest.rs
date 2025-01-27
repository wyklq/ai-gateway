use actix_cors::Cors;
use actix_web::middleware::Logger;
use actix_web::Scope as ActixScope;
use actix_web::{
    body::MessageBody,
    dev::{ServiceFactory, ServiceRequest, ServiceResponse},
    web::{self, Data},
    App, HttpServer,
};
use futures::{future::try_join, Future, TryFutureExt};
use langdb_ai_gateway::handler::chat::create_chat_completion;
use langdb_ai_gateway::handler::embedding::embeddings_handler;
use langdb_ai_gateway::handler::image::create_image;
use langdb_ai_gateway::handler::models::list_gateway_models;
use langdb_ai_gateway::handler::{AvailableModels, CallbackHandlerFn};
use langdb_ai_gateway::models::LlmModelDefinition;
use langdb_ai_gateway::otel::{TraceMap, TraceServiceImpl, TraceServiceServer};
use langdb_ai_gateway::types::gateway::CostCalculator;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

use crate::config::Config;
use crate::cost::DummyCostCalculator;
use crate::otel::DummyTraceWritterTransport;

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Clone)]
#[serde(crate = "serde")]
pub enum CorsOptions {
    Permissive,
    Custom(Vec<String>, usize),
}

#[derive(Error, Debug)]
pub enum ServerError {
    #[error(transparent)]
    Actix(#[from] std::io::Error),
    #[error(transparent)]
    Tonic(#[from] tonic::transport::Error),
}

#[derive(Clone, Debug)]
pub struct ApiServer {
    config: Config,
}

impl ApiServer {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub async fn start(self) -> Result<impl Future<Output = Result<(), ServerError>>, ServerError> {
        let trace_senders = Arc::new(TraceMap::new());
        let trace_senders_inner = Arc::clone(&trace_senders);

        let server = HttpServer::new(move || {
            let cors = Self::get_cors(CorsOptions::Permissive);
            Self::create_app_entry(
                cors,
                trace_senders_inner.clone(),
                self.config.models.clone(),
            )
        })
        .bind((self.config.rest.host.as_str(), self.config.rest.port))?
        .run()
        .map_err(ServerError::Actix);

        let writer = DummyTraceWritterTransport {};

        let trace_service = TraceServiceServer::new(TraceServiceImpl::new(
            Arc::new(TraceMap::new()),
            Box::new(writer),
        ));
        let tonic_server = tonic::transport::Server::builder().add_service(trace_service);
        let tonic_fut = tonic_server
            .serve("[::]:4317".parse().unwrap())
            .map_err(ServerError::Tonic);
        Ok(try_join(server, tonic_fut).map_ok(|_| ()))
    }

    fn create_app_entry(
        cors: Cors,
        trace_senders: Arc<TraceMap>,
        models: Vec<LlmModelDefinition>,
    ) -> App<
        impl ServiceFactory<
            ServiceRequest,
            Response = ServiceResponse<impl MessageBody>,
            Config = (),
            InitError = (),
            Error = actix_web::Error,
        >,
    > {
        let app = App::new();
        let callback = CallbackHandlerFn(None);

        app.wrap(Logger::default())
            .service(
                Self::attach_gateway_routes(web::scope("/v1"))
                    .app_data(Data::new(callback))
                    .app_data(web::Data::from(trace_senders.clone()))
                    .app_data(Data::new(AvailableModels(models)))
                    .app_data(Data::new(
                        Box::new(DummyCostCalculator {}) as Box<dyn CostCalculator>
                    )),
            )
            .wrap(cors)
    }

    fn get_cors(cors: CorsOptions) -> Cors {
        match cors {
            CorsOptions::Permissive => Cors::permissive(),
            CorsOptions::Custom(origins, max_age) => origins
                .into_iter()
                .fold(Cors::default(), |cors, origin| cors.allowed_origin(&origin))
                .max_age(max_age),
        }
    }

    fn attach_gateway_routes(scope: ActixScope) -> ActixScope {
        scope
            .route("/chat/completions", web::post().to(create_chat_completion))
            .route("/models", web::get().to(list_gateway_models))
            .route("/embeddings", web::post().to(embeddings_handler))
            .route("/images/generations", web::post().to(create_image))
    }
}

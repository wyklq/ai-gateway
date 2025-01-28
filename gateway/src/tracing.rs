use langdb_core::events::{self, BaggageSpanProcessor};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::TracerProvider;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Layer, Registry};

pub fn init_tracing() {
    let log_level = std::env::var("RUST_LOG").unwrap_or("info".to_string());
    let env_filter = EnvFilter::new(log_level);
    let color = std::env::var("ANSI_OUTPUT").map_or(true, |v| v == "true");

    // tracing syntax ->
    let builder = tracing_subscriber::fmt::layer()
        .pretty()
        .with_line_number(false)
        .with_file(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_ansi(color)
        .with_filter(env_filter);

    let otlp_exporter = opentelemetry_otlp::new_exporter().tonic();
    let provider = TracerProvider::builder()
        .with_span_processor(BaggageSpanProcessor::new([
            "langdb.parent_trace_id",
            "langdb.run_id",
            "langdb.label",
        ]))
        .with_batch_exporter(
            otlp_exporter.build_span_exporter().unwrap(),
            opentelemetry_sdk::runtime::Tokio,
        )
        .with_config(events::config())
        .build();
    let tracer = provider.tracer("langdb-ai-gateway");
    opentelemetry::global::set_tracer_provider(provider);

    let otel_layer = events::layer("langdb::user_tracing", LevelFilter::INFO, tracer);
    Registry::default()
        .with(builder)
        .with(otel_layer)
        .try_init()
        .expect("initialized subscriber successfully");
}

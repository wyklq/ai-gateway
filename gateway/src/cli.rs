use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(arg_required_else_help = true)]
pub struct Cli {
    /// Optional config file path
    #[arg(short, long, default_value = "config.yaml")]
    pub config: String,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Update the available models cache
    Update {
        /// Force update even if cache is fresh
        #[arg(short, long)]
        force: bool,
    },
    /// List all available models
    List,
    /// Start the API server (default if no command specified)
    Serve,
}

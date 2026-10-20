use situ::{app, cli, instance};

fn die(msg: &str) -> ! {
    eprintln!("situ: {msg}");
    std::process::exit(1)
}

fn main() {
    let cfg = match cli::parse(std::env::args().skip(1)) {
        Ok(cli::Cmd::Run(cfg)) => cfg,
        Ok(cli::Cmd::Help) => return print!("{}", cli::HELP),
        Err(e) => die(&e),
    };
    let _instance = instance::take_over().unwrap_or_else(|e| die(&e));
    instance::exit_on_suspend();
    let Err(e) = app::run(cfg);
    die(&e.to_string());
}

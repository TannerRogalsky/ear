fn main() -> anyhow::Result<()> {
    let args = <whisper::Args as clap::Parser>::parse();
    let segments = whisper::transcribe(&args)?;
    std::fs::write(args.output, serde_json::to_string(&segments)?)?;
    Ok(())
}

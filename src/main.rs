use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Debug, Clone)]
struct EntryArg {
    user: String,
    path: PathBuf,
}

impl std::str::FromStr for EntryArg {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        let mut parts = raw.splitn(2, '=');
        let user = parts
            .next()
            .ok_or_else(|| "entry missing user".to_string())?
            .trim();
        let path = parts
            .next()
            .ok_or_else(|| "entry missing path".to_string())?
            .trim();
        if user.is_empty() {
            return Err("entry user is empty".to_string());
        }
        if path.is_empty() {
            return Err("entry path is empty".to_string());
        }
        Ok(Self {
            user: user.to_string(),
            path: PathBuf::from(path),
        })
    }
}

#[derive(Parser, Debug, Clone)]
struct WhisperOptions {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny.en")]
    model: whisper::WhichModel,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    quantized: bool,

    /// Language.
    #[arg(long, default_value = "en")]
    language: Option<String>,

    /// Task, when no task is specified, the input tokens contain only the sot token which can
    /// improve things when in no-timestamp mode.
    #[arg(long, default_value_t = whisper::Task::Transcribe)]
    task: whisper::Task,

    /// Timestamps mode.
    #[arg(long, default_value_t = true)]
    timestamps: bool,

    /// Maximum initial timestamp index to consider.
    #[arg(long)]
    max_initial_timestamp_index: Option<u32>,

    /// Print the full DecodingResult structure rather than just the text.
    #[arg(long)]
    verbose: bool,
}

impl WhisperOptions {
    fn to_args(&self, input: PathBuf) -> whisper::Args {
        whisper::Args {
            cpu: self.cpu,
            model_id: self.model_id.clone(),
            revision: self.revision.clone(),
            model: self.model,
            input,
            seed: self.seed,
            tracing: self.tracing,
            quantized: self.quantized,
            language: self.language.clone(),
            task: self.task,
            timestamps: self.timestamps,
            max_initial_timestamp_index: self.max_initial_timestamp_index,
            verbose: self.verbose,
            output: PathBuf::new(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Transcribe and stitch multiple sources.")]
struct Args {
    /// Entries in the form "user=path".
    #[arg(long = "entry", value_name = "USER=PATH")]
    entries: Vec<EntryArg>,

    /// Output stitched transcript path.
    #[arg(long, default_value = "stitched.txt")]
    output: PathBuf,

    #[command(flatten)]
    whisper: WhisperOptions,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let entries = if args.entries.is_empty() {
        default_entries()
    } else {
        args.entries
    };

    let mut stitched_inputs = Vec::with_capacity(entries.len());
    for entry in entries {
        let whisper_args = args.whisper.to_args(entry.path);
        println!("Processing user: {}", entry.user);
        let segments = whisper::transcribe(&whisper_args)?;
        let stitched_segments = segments
            .into_iter()
            .map(|segment| stitch::Segment {
                start: segment.start,
                dr: stitch::SegmentDr {
                    text: segment.dr.text,
                },
            })
            .collect();
        stitched_inputs.push(stitch::Input::new(entry.user, stitched_segments));
    }

    let stitched = stitch::stitch(&stitched_inputs);
    stitch::write_stitched(&args.output, &stitched)?;
    Ok(())
}

fn default_entries() -> Vec<EntryArg> {
    vec![
        EntryArg {
            user: "Ceril".to_string(),
            path: PathBuf::from("./assets/1-jug_head.flac"),
        },
        EntryArg {
            user: "Red".to_string(),
            path: PathBuf::from("./assets/2-wutangtan.flac"),
        },
        EntryArg {
            user: "Kerben".to_string(),
            path: PathBuf::from("./assets/3-_kazoul.flac"),
        },
        EntryArg {
            user: "Domix".to_string(),
            path: PathBuf::from("./assets/4-gamingwolfplays.flac"),
        },
        EntryArg {
            user: "DM".to_string(),
            path: PathBuf::from("./assets/5-blue_tetris.flac"),
        },
        EntryArg {
            user: "Vokenar".to_string(),
            path: PathBuf::from("./assets/6-spiritguardian1.flac"),
        },
    ]
}

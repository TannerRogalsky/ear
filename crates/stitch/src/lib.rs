use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Input<T> {
    pub user: String,
    pub segments: Vec<T>,
}

impl<T> Input<T> {
    pub fn new(user: impl Into<String>, segments: Vec<T>) -> Self {
        Self {
            user: user.into(),
            segments,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InputFile {
    pub user: String,
    pub path: PathBuf,
}

impl InputFile {
    pub fn new(user: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            user: user.into(),
            path: path.into(),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Segment {
    pub start: f64,
    pub dr: SegmentDr,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SegmentDr {
    pub text: String,
}

#[derive(Debug)]
struct Entry {
    start: f64,
    order: usize,
    user: String,
    text: String,
}

pub fn read_segments(path: impl AsRef<Path>) -> Result<Vec<Segment>> {
    let path = path.as_ref();
    let data = std::fs::read(path).with_context(|| format!("read input file {}", path.display()))?;
    serde_json::from_slice(&data).with_context(|| format!("parse json {}", path.display()))
}

pub fn stitch_from_files(inputs: &[InputFile]) -> Result<String> {
    let mut collected = Vec::with_capacity(inputs.len());
    for input in inputs {
        let segments = read_segments(&input.path)?;
        collected.push(Input::new(input.user.clone(), segments));
    }
    Ok(stitch(&collected))
}

pub fn write_stitched(path: impl AsRef<Path>, stitched: &str) -> Result<()> {
    let path = path.as_ref();
    std::fs::write(path, stitched)
        .with_context(|| format!("write stitched output {}", path.display()))?;
    Ok(())
}

pub fn stitch(inputs: &[Input<Segment>]) -> String {
    let mut entries = Vec::new();
    let mut order = 0usize;
    for input in inputs {
        for segment in &input.segments {
            entries.push(Entry {
                start: segment.start,
                order,
                user: input.user.clone(),
                text: segment.dr.text.clone(),
            });
            order += 1;
        }
    }

    entries.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.order.cmp(&b.order))
    });

    let mut lines = Vec::new();
    let mut current_user: Option<String> = None;
    let mut current_text = String::new();
    for entry in entries {
        let cleaned = strip_timestamps(&entry.text);
        let mut entry_text = String::new();
        for line in cleaned.lines() {
            let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
            let line = line.trim();
            if line.is_empty() || line == "Thank you." {
                continue;
            }
            entry_text.push_str(line);
        }
        if entry_text.is_empty() {
            continue;
        }

        if let Some(user) = current_user.as_deref() {
            if user == entry.user.as_str() {
                current_text.push(' ');
                current_text.push_str(&entry_text);
            } else {
                lines.push(format!("{}: {}", user, current_text));
                current_user = Some(entry.user);
                current_text = entry_text;
            }
        } else {
            current_user = Some(entry.user);
            current_text = entry_text;
        }
    }
    if let Some(user) = current_user {
        lines.push(format!("{}: {}", user, current_text));
    }

    if lines.is_empty() {
        String::new()
    } else {
        let mut out = lines.join("\n");
        out.push('\n');
        out
    }
}

pub fn strip_timestamps(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            i += 2;
            while i + 1 < bytes.len() {
                if bytes[i] == b'|' && bytes[i + 1] == b'>' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

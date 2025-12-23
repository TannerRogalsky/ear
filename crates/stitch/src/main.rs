struct Input {
    user: &'static str,
    path: &'static str,
}

impl Input {
    fn new(user: &'static str, path: &'static str) -> Self {
        Self { user, path }
    }
}

fn main() {
    let files = [
        Input::new("Ceril", "./assets/1-jug_head.json"),
        Input::new("Red", "./assets/2-wutangtan.json"),
        Input::new("Kerben", "./assets/3-_kazoul.json"),
        Input::new("Domix", "./assets/4-gamingwolfplays.json"),
        Input::new("DM", "./assets/5-blue_tetris.json"),
        Input::new("Vokenar", "./assets/6-spiritguardian1.json"),
    ];

    let mut entries = Vec::new();
    let mut order = 0usize;
    for input in files {
        let data = std::fs::read(input.path).expect("read input file");
        let segments: Vec<Segment> = serde_json::from_slice(&data).expect("parse json");
        for segment in segments {
            entries.push(Entry {
                start: segment.start,
                order,
                user: input.user,
                text: segment.dr.text,
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

    let output = std::fs::File::create("stitched.txt").expect("create output file");
    let mut writer = std::io::BufWriter::new(output);
    let mut current_user: Option<&'static str> = None;
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

        match current_user {
            Some(user) if user == entry.user => {
                current_text.push(' ');
                current_text.push_str(&entry_text);
            }
            Some(user) => {
                use std::io::Write;
                writeln!(writer, "{}: {}", user, current_text).expect("write output");
                current_user = Some(entry.user);
                current_text = entry_text;
            }
            None => {
                current_user = Some(entry.user);
                current_text = entry_text;
            }
        }
    }
    if let Some(user) = current_user {
        use std::io::Write;
        writeln!(writer, "{}: {}", user, current_text).expect("write output");
    }
}

#[derive(serde::Deserialize)]
struct Segment {
    start: f64,
    dr: SegmentDr,
}

#[derive(serde::Deserialize)]
struct SegmentDr {
    text: String,
}

struct Entry {
    start: f64,
    order: usize,
    user: &'static str,
    text: String,
}

fn strip_timestamps(text: &str) -> String {
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

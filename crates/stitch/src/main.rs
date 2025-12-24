fn main() -> anyhow::Result<()> {
    let files = [
        stitch::InputFile::new("Ceril", "./assets/1-jug_head.json"),
        stitch::InputFile::new("Red", "./assets/2-wutangtan.json"),
        stitch::InputFile::new("Kerben", "./assets/3-_kazoul.json"),
        stitch::InputFile::new("Domix", "./assets/4-gamingwolfplays.json"),
        stitch::InputFile::new("DM", "./assets/5-blue_tetris.json"),
        stitch::InputFile::new("Vokenar", "./assets/6-spiritguardian1.json"),
    ];

    let stitched = stitch::stitch_from_files(&files)?;
    stitch::write_stitched("stitched.txt", &stitched)?;
    Ok(())
}

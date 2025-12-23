ffmpeg \
 -i ./assets/1-jug_head.flac \
 -i ./assets/2-wutangtan.flac \
 -i ./assets/3-_kazoul.flac \
 -i ./assets/4-gamingwolfplays.flac \
 -i ./assets/5-blue_tetris.flac \
 -i ./assets/6-spiritguardian1.flac \
 -filter_complex "amix=inputs=6:normalize=0" \
 -c:a libvorbis -q:a 5 \
 mixed.ogg
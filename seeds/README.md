# seeds

Sample data for bringing the stand to life.

`audio/` holds call recordings for `make seed`, which uploads them to the
`raw-files` bucket and records one row per file in `analysis.calls`. The file
stem becomes the `call_id`, so re-running is safe: the same file seeds the same
call rather than a new one.

**The contents of `audio/` are not versioned, deliberately.** Recordings of real
sales calls are personal data — voices, names, phone numbers, sometimes payment
details — and anything committed here would stay in git history permanently,
including in any copy of the repository shared later. Put files in place locally,
or on the deploy host under the same path.

Supported: mp3, wav, m4a, ogg, flac — anything ffmpeg opens, since duration is
read with pydub.

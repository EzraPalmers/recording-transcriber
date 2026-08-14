---
name: gather-notes
description: Turn this repository's transcripts into a sourced, timestamped note under notes/. Use when asked to make notes from a recording or a transcript, to summarise or write up what was said, to pull out the decisions, questions, examples or key points, to find every mention of a topic across recordings, or to recover the timestamps for something that was said.
---

# Gather notes from transcripts

The tool writes each recording twice: plain text at
`transcripts/<subfolders>/<name>.txt` and the same segments timestamped at
`transcripts_timestamps/<subfolders>/<name>.txt`, both mirroring the folder
layout under `recordings/`. Read the plain tree, then recover timestamps from
the matching file in the timestamped tree. Both trees are gitignored, so treat
their contents as local input, never as something to commit.

A recording can be anything the tool accepts: a talk, an interview, a podcast
episode, a voice memo, a lecture. The request names a topic, and may name a
subject or folder-like word alongside it. That word is a topic, not a filter:
fuzzy-match it against the folder names under `transcripts/` to decide where to
look first and where the note belongs, but do not treat a non-matching folder
as excluded. How recordings are grouped and named is the user's choice; read it
as a hint, never as a rule about what kind of recording this is.

## Output shape

The default is a sourced note: exact quotes for the wording that matters, every
entry carrying its recording and timestamp range. Adapt to what was asked. A
summary, strict notes, and an extract of the relevant passages are all
legitimate outputs, and a request for one of them overrides the default. In
every shape, sourcing stays: no claim without the recording and timestamp it
came from.

Faithful paraphrase is allowed where it is clearer than the raw wording. Do not
coin a label, rename a concept, or introduce shorthand that is not in the
transcript; where the speaker had no name for something, describe it
structurally. Reserve quotation for wording that matters.

## Method

1. Inspect the transcript tree before reading. List `transcripts/` and match
   any named folder against what is actually there. Never abbreviate, shorten,
   or otherwise rewrite a folder or file name when reproducing it later. Two
   inputs sharing a stem keep their extensions in the output name, so
   `pilot.mp4.txt` is a normal transcript name, not a mistake.
2. For a named recording, read its plain transcript end to end in chunks. It is
   the only way to follow pronouns and topic changes.
3. For a topic across many recordings, search first, then read. Grep the plain
   transcripts for each term and for the synonyms and spellings a speaker would
   plausibly use. Match counts are triage only: they order the reading, they do
   not measure relevance. Read each hit's complete surrounding discussion, from
   where the topic starts to where it changes, not a fixed number of context
   lines; Whisper writes detected speech segments rather than sentences, so a
   fixed window cuts discussions in half. Then search again on the distinctive
   terms and the plausible mishearings the first pass revealed.
4. Organise by topic rather than by file. Merge overlapping coverage from
   several recordings into one entry and record every source it came from.
   Choose the sections the content supports: what was defined, with arguments,
   examples and caveats; questions raised, themes and disagreements; decisions
   and next steps; passages that need an audio check. Do not force a section
   the recording does not fill.
5. Recover timestamps from `transcripts_timestamps/` for the same files.
   Timestamps appear as `[hh:mm:ss -> hh:mm:ss]` at the start of each segment.
   Search that file for the exact wording extracted. A passage crossing a
   segment boundary will not match as one string: search the opening wording
   and the closing wording separately and take the first segment's start and
   the last segment's end. Only if exact matching fails, fall back to position:
   the two files hold the same segments in the same order, so line N of one is
   line N of the other.
6. Write the note to `notes/`, mirroring the transcript subfolders. A note
   drawn from one folder goes to `notes/<same subfolders>/<topic-slug>.md`; a
   note spanning several goes to `notes/<topic-slug>.md` with one section per
   source folder. Create the parent directories; on a fresh clone only `notes/`
   itself exists. The topic slug is the topic lowercased with spaces and
   punctuation replaced by hyphens, keeping only `a-z`, `0-9` and `-`, so
   `Bias/variance: what?` becomes `bias-variance-what`. If a file of that name
   already exists, stop and ask whether to replace it, add to it, or write
   under another name.
7. Open the note with a source table, one row per transcript: the transcript
   path, the timestamp ranges it covers, and an ASR warning where the audio was
   unclear. Then give the material with its timestamps inline.
8. Report the path written and the source table. Offer to widen the search
   terms, to widen the reading to more folders, and to redo the note in another
   shape, a summary or an extract only.

## Constraints

- Extract from the two transcript trees. Do not read existing notes under
  `notes/` and recycle them; the point is fresh material traced to a source and
  a time. Read one only when explicitly told to.
- The transcripts are ASR output with no speaker labels, and names, jargon, and
  acronyms are often misheard. Treat a transcript line as a lead rather than a
  reliable quotation. Flag a garbled passage as needing an audio check instead
  of quietly repairing it into something that may never have been said, and
  never invent a speaker label.
- Do not run the transcriber to fill a gap. Transcription is a heavy run and is
  the user's to start.
- Missing input does not stop the job. If a recording has no transcript, or a
  plain transcript has no timestamped counterpart, continue with what exists
  and list the gaps prominently in the note and in the report: name the missing
  file, and mark any entry that could not be given a timestamp. Never present a
  note built over a gap as complete coverage.
- Report plainly rather than working around: a named folder with no match under
  `transcripts/`, a topic with no match anywhere, and an unreadable file.

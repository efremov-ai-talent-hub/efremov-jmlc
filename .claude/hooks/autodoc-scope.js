// PreToolUse (Bash): supplies the agent with the actual contents of a push.
//
// Why. The autodoc gate denies `git push` with "record ONE entry summarizing this
// whole update". It never says what "update" spans, and the only context an agent
// has is its own session — so it documents its own work, while a push routinely
// carries commits written by other sessions.
//
// The gate's wording is not ours to change, so instead of rewriting it we put the
// missing facts next to it: the commit list from the same range the gate uses, and
// an explicit statement that other authors' commits are in scope too.
//
// This hook never denies — blocking stays the gate's job. It only adds context.
//
// Known gaps, deliberately not handled: `pushd`, a subshell `cd` leaking forward,
// short-circuited branches, `bash -cx`.

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

let cmd = "";
try {
  cmd = (JSON.parse(fs.readFileSync(0, "utf8").replace(/^﻿/, "")).tool_input || {}).command || "";
} catch {}
if (!cmd.trim()) process.exit(0);

// Global git options may sit between `git` and the subcommand. Anchoring on this
// prefix is what separates a real push from a command that merely mentions one:
// a loose /git.*push/ also fires on `git log --grep=push` and on commit messages
// containing the word, injecting an unrelated commit list as if it were fact.
const G = "git\\s+(?:(?:-C|-c)\\s+\\S+\\s+|--(?:git-dir|work-tree|exec-path)(?:=\\S+|\\s+\\S+)\\s+)*";

// A nested shell hides its own `cd` from us (`bash -c "cd sub && git push"`),
// so any repo we resolved out here would be a guess. Silence beats a wrong list.
if (/\b(?:ba|z|k)?sh\s+-[a-z]*c\b/.test(cmd)) process.exit(0);

// Test each segment for STARTING with git…push, rather than testing the whole line
// for containing it. `echo "will git push later"` contains it; that segment starts
// with echo, so it is not a push.
const PUSH = new RegExp("^[\\s(]*(?:[A-Za-z_][A-Za-z0-9_]*=\\S*\\s+)*" + G + "push\\b");
const ARG = "(\"[^\"]+\"|'[^']+'|\\S+)";
const segments = cmd.split(/\|\||&&|[;|\n]/);
const pushAt = segments.findIndex((s) => PUSH.test(s));
if (pushAt < 0) process.exit(0);

// Retarget to the repo actually being pushed. Only the pushing segment's own -C
// counts — `git -C other status && git push` pushes here, not there. A bare `cd`,
// by contrast, moves the shell for every later segment, so take the LAST one
// before the push rather than the first.
let target = (segments[pushAt].match(new RegExp("(?:-C|--git-dir=?)\\s*" + ARG)) || [])[1];
if (!target) {
  for (let i = pushAt - 1; i >= 0; i--) {
    const m = segments[i].match(new RegExp("^[\\s(]*cd\\s+" + ARG));
    if (m) { target = m[1]; break; }
  }
}
if (target) {
  let dir = target.replace(/^['"]|['"]$/g, "");
  // `--git-dir=/repo/.git` points at the metadata directory, not the work tree.
  if (/(^|\/)\.git\/?$/.test(dir)) dir = path.dirname(dir.replace(/\/$/, ""));
  try {
    process.chdir(fs.statSync(dir).isDirectory() ? dir : path.dirname(dir));
  } catch { process.exit(0); }   // cannot resolve the target — stay silent rather than guess
}

const sh = (c) => {
  try { return execSync(c, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); }
  catch { return ""; }
};

// Same range the gate computes: commits on HEAD absent from every remote-tracking
// branch. Needs no upstream configured.
const RANGE = "HEAD --not --remotes";
const CAP = 25;

const total = parseInt(sh(`git rev-list --count ${RANGE}`), 10);
if (!total) process.exit(0);

// Capped: with no remote-tracking refs (first push to a new remote, fresh clone
// before fetch) the range degenerates to the entire history.
// Format string MUST stay quoted: unquoted, the shell splits it on spaces and dies
// on the parens, sh() swallows the error and the hook silently reports nothing.
const log = sh(`git log ${RANGE} --max-count=${CAP} --format='%h %s (%an)'`);
if (!log) process.exit(0);

const shown = log.split(/\r?\n/).filter(Boolean).length;
// Authors over the FULL range, not just the shown slice: past the cap a multi-author
// range would otherwise look single-author, dropping the one line that most directly
// refutes "this push is my work".
const authors = new Set(sh(`git log ${RANGE} --format='%an'`).split(/\r?\n/).filter(Boolean));

const lines = [
  `This push contains ${total} commit(s)${shown < total ? `, first ${shown} shown` : ""}:`,
  log,
  shown < total ? `… and ${total - shown} more (git log ${RANGE} --oneline)` : "",
  "",
  "IF THE AUTODOC GATE FIRED, the entry must cover the WHOLE list above, not your own",
  "work. Its \"this whole update\" means the entire push. This working tree is shared by",
  "several sessions, so some commits are likely not yours — they are in scope too. Read",
  "them (git show <sha>) before writing; your memory of this session does not cover them.",
  "ONE entry for the whole range, not one per commit.",
  "Name that commit `docs: changelog for <short summary of the push>` — the gate offers",
  "a bare `docs: changelog` as its example, which makes every such commit in the log",
  "indistinguishable from the last one. Ignore the gate's example, keep the prefix.",
  authors.size > 1 ? `Authors here: ${[...authors].join(", ")} — so the commits are demonstrably not all yours.` : "",
  "",
  `Range is \`${RANGE}\` — the same one the gate uses. For an explicit refspec`,
  "(git push origin feature:main) or a detached HEAD the real payload may differ."
];

process.stdout.write(JSON.stringify({
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    additionalContext: lines.filter((l, i) => l !== "" || i === 0 || lines[i - 1] !== "").join("\n")
  }
}));
process.exit(0);

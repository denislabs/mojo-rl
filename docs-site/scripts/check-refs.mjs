#!/usr/bin/env node
/**
 * check-refs — fail the build when the docs cite a repo path that does not exist.
 *
 * Motivation: three of six example paths cited in README.md / CLAUDE.md were
 * already stale when the docs site was started (see docs/DOC_SITE_PLAN.md §7).
 * Documentation that names files rots silently; this makes it rot loudly.
 *
 * Scans every .md/.mdx page under src/content/docs for repo-relative paths
 * (examples/…, tests/…, benchmarks/…, mojo_rl/…, docs/…) appearing in inline
 * code, links, or code fences, and checks each one against the repository.
 *
 * Usage:  node scripts/check-refs.mjs
 * Exit:   0 = all references resolve, 1 = at least one is broken.
 */

import { readdir, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, relative } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(SITE_ROOT, '..');
const DOCS_DIR = join(SITE_ROOT, 'src/content/docs');

/**
 * Top-level repo directories whose paths we validate when mentioned.
 *
 * `docs-site` must come first: the regex takes the leftmost match, so without
 * it a path like `docs-site/scripts/build-media.mjs` matches only its
 * `scripts/…` tail and is then resolved against the repo root, where it does
 * not exist.
 */
const TRACKED = [
	'docs-site',
	'examples',
	'tests',
	'benchmarks',
	'mojo_rl',
	'docs',
	'gifs',
	'scripts',
];

/**
 * Matches a repo-relative file path with an extension.
 * Deliberately requires an extension: bare directory mentions ("examples/") are
 * prose, not references, and produce noise.
 */
const PATH_RE = new RegExp(
	String.raw`\b(?:${TRACKED.join('|')})\/[\w./@-]+\.[a-zA-Z0-9]+`,
	'g',
);

/** Paths that are illustrative rather than real (e.g. "my_env.mojo" in a how-to). */
const ALLOWLIST = new Set([
	// add deliberate placeholders here, with a comment saying why
]);

async function* walk(dir) {
	for (const entry of await readdir(dir, { withFileTypes: true })) {
		const full = join(dir, entry.name);
		if (entry.isDirectory()) yield* walk(full);
		else if (/\.mdx?$/.test(entry.name)) yield full;
	}
}

function lineOf(content, index) {
	return content.slice(0, index).split('\n').length;
}

const broken = [];
let checked = 0;
let pages = 0;

for await (const file of walk(DOCS_DIR)) {
	pages++;
	const content = await readFile(file, 'utf8');
	const seen = new Set();

	for (const match of content.matchAll(PATH_RE)) {
		const path = match[0];
		// Report each distinct path once per page.
		const key = `${file}::${path}`;
		if (seen.has(key)) continue;
		seen.add(key);

		if (ALLOWLIST.has(path)) continue;
		checked++;

		if (!existsSync(join(REPO_ROOT, path))) {
			broken.push({
				page: relative(SITE_ROOT, file),
				line: lineOf(content, match.index),
				path,
			});
		}
	}
}

if (broken.length === 0) {
	console.log(`check-refs: ${checked} repo references across ${pages} pages — all resolve.`);
	process.exit(0);
}

console.error(`\ncheck-refs: ${broken.length} broken reference(s):\n`);
for (const b of broken) {
	console.error(`  ${b.page}:${b.line}`);
	console.error(`    ${b.path}  → not found in ${REPO_ROOT}\n`);
}
console.error(
	'Fix the path, or add it to ALLOWLIST in scripts/check-refs.mjs if it is\n' +
		'a deliberate placeholder.\n',
);
process.exit(1);

#!/usr/bin/env node
/**
 * Post-build guard: every page's social-card metadata must actually resolve.
 *
 * Exists because a real bug shipped silently — Starlight reports the site root's
 * route id as `''` while its content-collection id is `index`, so the homepage
 * pointed at `/og/.png` and shares of the landing page rendered a broken image.
 * Nothing failed; the tags were present and merely wrong.
 *
 * Checks, per page:
 *   - og:image is present and absolute (relative URLs are rejected by crawlers)
 *   - the file it names exists in dist/
 *   - twitter:card is summary_large_image only when an image really exists
 *   - the JSON-LD block parses
 * Plus: no generated card is left unreferenced.
 *
 * Runs after `astro build`. Exits non-zero on any failure.
 */
import { readFileSync, existsSync } from 'node:fs';
import { globSync } from 'node:fs';
import { join, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const ORIGIN = 'https://mojo-rl.denislabs.com';

if (!existsSync(dist)) {
	console.error('check-og: dist/ not found — run `astro build` first.');
	process.exit(1);
}

const pages = globSync('**/index.html', { cwd: dist });
const failures = [];
const referenced = new Set();

for (const page of pages) {
	const html = readFileSync(join(dist, page), 'utf8');
	const head = html.split('</head>')[0];
	const route = '/' + page.replace(/index\.html$/, '');

	const og = head.match(/<meta property="og:image" content="([^"]+)"/);
	if (!og) {
		failures.push(`${route} — no og:image`);
		continue;
	}

	const url = og[1];
	if (!url.startsWith('http')) {
		failures.push(`${route} — og:image is relative (${url}); crawlers reject it`);
		continue;
	}
	if (!url.startsWith(ORIGIN)) {
		failures.push(`${route} — og:image origin is not ${ORIGIN} (${url})`);
		continue;
	}

	const rel = url.slice(ORIGIN.length);
	referenced.add(rel);
	if (!existsSync(join(dist, rel))) {
		failures.push(`${route} — og:image 404s: ${rel}`);
	}

	if (/twitter:card" content="summary_large_image"/.test(head) && !url) {
		failures.push(`${route} — declares summary_large_image with no image`);
	}

	const ld = head.match(/<script type="application\/ld\+json">(.*?)<\/script>/s);
	if (!ld) {
		failures.push(`${route} — no JSON-LD`);
	} else {
		try {
			JSON.parse(ld[1].replace(/&quot;/g, '"').replace(/&amp;/g, '&'));
		} catch (e) {
			failures.push(`${route} — JSON-LD does not parse: ${e.message}`);
		}
	}
}

const orphans = globSync('og/**/*.png', { cwd: dist })
	.map((f) => '/' + f)
	.filter((f) => !referenced.has(f));

for (const o of orphans) failures.push(`orphan card, generated but never referenced: ${o}`);

if (failures.length) {
	console.error(`check-og: ${failures.length} problem(s) across ${pages.length} pages\n`);
	for (const f of failures) console.error('  ✗ ' + f);
	process.exit(1);
}

console.log(
	`check-og: ${pages.length} pages — og:image resolves, JSON-LD parses, no orphan cards.`
);

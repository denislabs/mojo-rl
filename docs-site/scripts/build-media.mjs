#!/usr/bin/env node
/**
 * build-media — transcode the repo's demo GIFs into web-deliverable video.
 *
 * `gifs/` holds ~110 MB of animated GIFs of trained agents and playable demos.
 * Shipping those raw would dominate the page weight of the whole site
 * (half_cheetah_sac_trained.gif alone is 22 MB). H.264 in an MP4 container is
 * typically 20-50x smaller for this kind of flat-colour screen capture, and
 * every browser plays it.
 *
 * For each GIF this writes, into public/media/:
 *   <name>.mp4    H.264, yuv420p, faststart, loopable
 *   <name>.webp   poster frame (first frame), shown before play and when the
 *                 viewer prefers reduced motion
 *
 * Output is committed so a clone can build the site without ffmpeg.
 * Re-run after adding a GIF; existing outputs are skipped unless --force.
 *
 * Usage:  node scripts/build-media.mjs [--force]
 */

import { execFileSync } from 'node:child_process';
import { readdirSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, basename } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(SITE_ROOT, '..');
const SRC_DIR = join(REPO_ROOT, 'gifs');
const OUT_DIR = join(SITE_ROOT, 'public/media');

// ffmpeg ships in the pixi environments rather than on PATH.
const FFMPEG_CANDIDATES = [
	join(REPO_ROOT, '.pixi/envs/cpu/bin/ffmpeg'),
	join(REPO_ROOT, '.pixi/envs/default/bin/ffmpeg'),
	'ffmpeg',
];

const force = process.argv.includes('--force');

function findFfmpeg() {
	for (const c of FFMPEG_CANDIDATES) {
		try {
			execFileSync(c, ['-version'], { stdio: 'ignore' });
			return c;
		} catch {
			/* try the next one */
		}
	}
	console.error(
		'build-media: no ffmpeg found. Tried:\n  ' + FFMPEG_CANDIDATES.join('\n  '),
	);
	process.exit(1);
}

const ffmpeg = findFfmpeg();
mkdirSync(OUT_DIR, { recursive: true });

const gifs = readdirSync(SRC_DIR).filter((f) => f.endsWith('.gif'));
let srcBytes = 0;
let outBytes = 0;
let built = 0;
let skipped = 0;

for (const gif of gifs.sort()) {
	const name = basename(gif, '.gif');
	const src = join(SRC_DIR, gif);
	const mp4 = join(OUT_DIR, `${name}.mp4`);
	const poster = join(OUT_DIR, `${name}.webp`);

	srcBytes += statSync(src).size;

	if (!force && existsSync(mp4) && existsSync(poster)) {
		outBytes += statSync(mp4).size + statSync(poster).size;
		skipped++;
		continue;
	}

	// Two caps, both about fitness for purpose rather than only file size:
	//
	//   MAX_WIDTH   the widest these ever render in the content column is ~700px,
	//               so 720p captures (half_cheetah) are pure waste.
	//   MAX_SECONDS these are loops illustrating a behaviour. The source GIFs
	//               run up to 150s (lunar_lander_ppo, 5000 frames), which is a
	//               film, not a demo. ~12s shows the gait and loops cleanly.
	//
	// H.264 also needs even dimensions, hence the -2 height.
	const MAX_WIDTH = 960;
	const MAX_SECONDS = 12;
	const scale = `scale='min(${MAX_WIDTH},iw)':-2`;

	execFileSync(
		ffmpeg,
		[
			'-y', '-loglevel', 'error',
			'-i', src,
			'-t', String(MAX_SECONDS),
			'-movflags', '+faststart',
			'-pix_fmt', 'yuv420p',
			'-vf', scale,
			'-c:v', 'libx264',
			'-preset', 'slow',
			'-crf', '30',
			'-an',
			mp4,
		],
		{ stdio: 'inherit' },
	);

	execFileSync(
		ffmpeg,
		[
			'-y', '-loglevel', 'error',
			'-i', src,
			'-vframes', '1',
			'-vf', scale,
			'-quality', '80',
			poster,
		],
		{ stdio: 'inherit' },
	);

	outBytes += statSync(mp4).size + statSync(poster).size;
	built++;
	const mb = (n) => (n / 1048576).toFixed(1);
	console.log(
		`  ${name.padEnd(34)} ${mb(statSync(src).size).padStart(6)} MB → ` +
			`${mb(statSync(mp4).size).padStart(5)} MB`,
	);
}

const mb = (n) => (n / 1048576).toFixed(1);
console.log(
	`\nbuild-media: ${built} built, ${skipped} cached — ` +
		`${mb(srcBytes)} MB of GIF → ${mb(outBytes)} MB of MP4+poster ` +
		`(${(srcBytes / outBytes).toFixed(1)}x smaller).`,
);

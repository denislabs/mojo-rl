/**
 * Builds src/assets/noeira-lockup-og.png — the moon mark + "mojo-rl" wordmark
 * used as the logo on every Open Graph card.
 *
 * Run manually after a brand change:
 *   node scripts/build-brand-lockup.mjs
 *
 * Why this exists rather than an SVG <text> element: sharp's SVG renderer uses
 * *system* fonts and falls back **silently** when one is missing. Manrope is not
 * installed system-wide, so `font-family="Manrope"` renders as Helvetica and
 * looks subtly off-brand with no error. Converting the text to outlines with
 * opentype.js removes the font dependency entirely — the PNG is deterministic
 * on any machine.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import opentype from 'opentype.js';
import sharp from 'sharp';

const root = fileURLToPath(new URL('..', import.meta.url));

const FONT = `${root}src/assets/fonts/Manrope-ExtraBold.ttf`;
const MARK = `${root}src/assets/noeira-mark.svg`;
const OUT = `${root}src/assets/noeira-lockup-og.png`;

const TEXT = 'mojo-rl';
const FONT_SIZE = 96;
const MARK_SIZE = 104;
const GAP = 26;
/** Rendered at 2× then downscaled, so the outlines stay crisp on the card. */
const SCALE = 2;

const font = opentype.parse(new Uint8Array(readFileSync(FONT)).buffer);

// Baseline is placed so the x-height centres against the mark.
const path = font.getPath(TEXT, 0, FONT_SIZE * 0.74, FONT_SIZE);
const { x2: textWidth } = path.getBoundingBox();

// +2px so the final glyph's right edge is not clipped by the raster bounds.
const width = Math.ceil(MARK_SIZE + GAP + textWidth) + 2;
const height = MARK_SIZE;

const markSvg = readFileSync(MARK, 'utf8');
const markPng = await sharp(Buffer.from(markSvg), { density: 900 })
	.resize(MARK_SIZE * SCALE, MARK_SIZE * SCALE, {
		fit: 'contain',
		background: { r: 0, g: 0, b: 0, alpha: 0 },
	})
	.png()
	.toBuffer();

// Wordmark in --sl-color-white on transparent, so it reads on the ink canvas.
const textSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width * SCALE}" height="${height * SCALE}" viewBox="0 0 ${width} ${height}">
	<g transform="translate(${MARK_SIZE + GAP}, ${(height - FONT_SIZE) / 2})">
		<path d="${path.toPathData(3)}" fill="#ffffff"/>
	</g>
</svg>`;

const info = await sharp(Buffer.from(textSvg))
	.composite([{ input: markPng, top: 0, left: 0 }])
	.png()
	.toFile(OUT);

console.log(
	`brand lockup: ${info.width}×${info.height}, ${(info.size / 1024).toFixed(1)} KiB → ${OUT.replace(root, '')}`
);

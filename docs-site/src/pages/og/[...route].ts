/**
 * Per-page Open Graph card generator.
 *
 * Emits /og/<page-id>.png for every docs page at build time. The <meta> tags
 * that point at these live in src/components/Head.astro.
 *
 * Fonts are vendored under src/assets/fonts rather than fetched from Google at
 * build time: canvaskit needs TTF (fontsource ships woff2 only), and a network
 * dependency in the build would make CI flaky. Both are OFL — licences sit
 * beside the files.
 */
import { getCollection } from 'astro:content';
import { OGImageRoute } from 'astro-og-canvas';

const docs = await getCollection('docs');

const pages = Object.fromEntries(docs.map((page) => [page.id, page.data]));

/** noeira ink scale + accents, as canvaskit RGB triples. */
const INK_900: [number, number, number] = [14, 17, 23];
const INK_950: [number, number, number] = [9, 12, 18];
const MOON_CYAN: [number, number, number] = [53, 208, 224];
const TEXT: [number, number, number] = [255, 255, 255];
const TEXT_MUTED: [number, number, number] = [150, 163, 184];

// OGImageRoute is async — without the await this destructures a Promise and
// Astro fails the build with "getStaticPaths() function is required".
// The route param is derived from the filename, so there is no `param` option.
export const { getStaticPaths, GET } = await OGImageRoute({
	pages,
	getImageOptions: (_path, page: (typeof pages)[string]) => ({
		title: page.title,
		description: page.description,

		// Light-on-transparent lockup: the card is always dark, regardless of the
		// viewer's theme. The 2x PNG rather than the SVG because canvaskit
		// rasterises bitmaps only.
		logo: { path: './src/assets/mojo-rl-logo-transparent-dark-2x.png', size: [300] },

		bgGradient: [INK_900, INK_950],
		border: { color: MOON_CYAN, width: 16, side: 'inline-start' },
		padding: 72,

		font: {
			title: {
				color: TEXT,
				size: 64,
				weight: 'ExtraBold',
				lineHeight: 1.15,
				families: ['Manrope'],
			},
			description: {
				color: TEXT_MUTED,
				size: 30,
				weight: 'Medium',
				lineHeight: 1.4,
				families: ['Manrope'],
			},
		},

		fonts: [
			'./src/assets/fonts/Manrope-ExtraBold.ttf',
			'./src/assets/fonts/Manrope-Medium.ttf',
			'./src/assets/fonts/JetBrainsMono-SemiBold.ttf',
		],

		format: 'PNG',
	}),
});

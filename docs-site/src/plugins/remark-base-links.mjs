/**
 * Prefixes Astro's `base` onto root-relative Markdown links and images.
 *
 * The site is served under `/docs` (noeira.ai keeps its root for the marketing
 * site), and Starlight applies `base` to its sidebar but not to links written
 * in content. Pages keep writing `[monitor](/tooling/monitor/)`, which stays
 * true to the docs' own tree whatever the mount point is; this plugin turns
 * it into `/docs/tooling/monitor/` at build time.
 *
 * Only `/x` is rewritten: protocol-relative `//host`, absolute URLs, `#anchors`
 * and relative paths pass through, as does anything already under the base.
 */
export default function remarkBaseLinks({ base }) {
	const prefix = base.replace(/\/+$/, '');
	const rewrite = (url) =>
		typeof url === 'string' &&
		url.startsWith('/') &&
		!url.startsWith('//') &&
		url !== prefix &&
		!url.startsWith(prefix + '/')
			? prefix + url
			: url;

	const walk = (node) => {
		if (node.type === 'link' || node.type === 'image' || node.type === 'definition') {
			node.url = rewrite(node.url);
		}
		if (node.children) node.children.forEach(walk);
	};

	return (tree) => walk(tree);
}

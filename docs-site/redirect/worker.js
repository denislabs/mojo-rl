/**
 * mojo-rl.denislabs.com -> noeira.ai/docs, permanently, path and query kept.
 *
 * Deployed under the OLD Worker's name (mojo-rl-docs), so it replaces the
 * static site on that hostname instead of competing with it for the domain.
 * Shared links and search results keep working: /start/why/ lands on
 * /docs/start/why/.
 */
export default {
	fetch(request) {
		const url = new URL(request.url);
		return Response.redirect(`https://noeira.ai/docs${url.pathname}${url.search}`, 301);
	},
};

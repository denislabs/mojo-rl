// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	// Canonical origin. Required by @astrojs/sitemap (bundled with Starlight),
	// which silently skips without it, and used for canonical <link> tags.
	site: 'https://mojo-rl.denislabs.com',
	integrations: [
		starlight({
			title: 'mojo-rl',
			description:
				'A reinforcement learning framework written in Mojo — 40+ algorithms, a deep learning framework, 2D/3D physics engines, and 25 native environments.',
			logo: {
				// Full lockup (mark + wordmark), so it replaces the title text.
				// Starlight's `dark`/`light` name the THEME the file is shown in:
				// `dark` needs light-on-transparent, `light` needs dark-on-transparent.
				// Wordmarks are outlined paths — no font dependency.
				dark: './src/assets/mojo-rl-logo-transparent-dark.svg',
				light: './src/assets/mojo-rl-logo-light.svg',
				alt: 'mojo-rl',
				replacesTitle: true,
			},
			favicon: '/favicon.svg',
			head: [{ tag: 'meta', attrs: { name: 'theme-color', content: '#0E1117' } }],
			customCss: ['./src/styles/noeira.css'],
			// Adds og:image + JSON-LD on top of Starlight's own head tags.
			components: { Head: './src/components/Head.astro' },
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/denislabs/mojo-rl',
				},
			],
			expressiveCode: {
				themes: ['github-dark-default', 'github-light'],
				styleOverrides: {
					borderRadius: '0.625rem',
					borderColor: 'var(--sl-color-hairline)',
					codeFontFamily: 'var(--sl-font-mono)',
					uiFontFamily: 'var(--sl-font)',
					codeBackground: 'var(--noeira-surface)',
					frames: {
						editorTabBarBackground: 'var(--noeira-surface-raised)',
						editorActiveTabIndicatorBottomColor: 'var(--sl-color-text-accent)',
						terminalTitlebarBackground: 'var(--noeira-surface-raised)',
						terminalBackground: 'var(--noeira-surface)',
					},
				},
			},
			// Nav mirrors docs/DOC_SITE_PLAN.md §5. Sections are added as their
			// phase lands, rather than shipping empty placeholder groups.
			sidebar: [
				{
					label: 'Start here',
					items: [
						{ label: 'Why mojo-rl', slug: 'start/why' },
						{ label: 'Installation', slug: 'start/installation' },
						{ label: 'Quickstart: tabular', slug: 'start/quickstart-tabular' },
						{ label: 'Quickstart: GPU training', slug: 'start/quickstart-gpu' },
						{ label: 'RL in five minutes', slug: 'start/rl-in-five-minutes' },
					],
				},
				{
					label: 'Concepts',
					items: [
						{ label: 'The layered stack', slug: 'concepts/architecture' },
						{ label: 'Compile-time composition', slug: 'concepts/compile-time-composition' },
						{ label: 'Traits', slug: 'concepts/traits' },
						{ label: 'CPU and GPU paths', slug: 'concepts/cpu-gpu' },
						{ label: 'Checkpoints', slug: 'concepts/checkpoints' },
					],
				},
				{
					label: 'Environments',
					items: [
						{ label: 'Overview', slug: 'environments' },
						{ label: 'Tabular', slug: 'environments/tabular' },
						{ label: 'Classic control', slug: 'environments/classic-control' },
						{ label: '2D physics', slug: 'environments/2d-physics' },
						{ label: '3D locomotion', slug: 'environments/3d-locomotion' },
						{ label: '3D manipulation', slug: 'environments/3d-manipulation' },
						{ label: 'Arcade games', slug: 'environments/arcade' },
						{ label: 'Atari 2600', slug: 'environments/atari' },
						{ label: 'Board games', slug: 'environments/board-games' },
						{ label: 'Procgen & Craftax', slug: 'environments/procgen' },
						{ label: 'Gymnasium wrappers', slug: 'environments/gymnasium' },
						{ label: 'Writing your own', slug: 'environments/custom' },
					],
				},
				{
					label: 'Algorithms',
					items: [
						{ label: 'Overview', slug: 'algorithms' },
						{
							label: '1 · Tabular and linear',
							collapsed: true,
							items: [
								{ label: 'TD methods', slug: 'algorithms/tabular/td-methods' },
								{ label: 'Function approximation', slug: 'algorithms/tabular/function-approximation' },
								{ label: 'Classical policy gradient', slug: 'algorithms/tabular/policy-gradient' },
							],
						},
						{
							label: '2 · Value-based',
							collapsed: true,
							items: [
								{ label: 'DQN', slug: 'algorithms/dqn' },
								{ label: 'C51 and Rainbow', slug: 'algorithms/rainbow' },
							],
						},
						{
							label: '3 · Policy gradient',
							collapsed: true,
							items: [
								{ label: 'A2C', slug: 'algorithms/a2c' },
								{ label: 'PPO', slug: 'algorithms/ppo' },
							],
						},
						{
							label: '4 · Continuous control',
							collapsed: true,
							items: [
								{ label: 'DDPG', slug: 'algorithms/ddpg' },
								{ label: 'TD3', slug: 'algorithms/td3' },
								{ label: 'SAC', slug: 'algorithms/sac' },
								{ label: 'REDQ', slug: 'algorithms/redq' },
							],
						},
						{
							label: '5 · Model-based',
							collapsed: true,
							items: [
								{ label: 'MBPO', slug: 'algorithms/mbpo' },
								{ label: 'TD-MPC2', slug: 'algorithms/tdmpc2' },
								{ label: 'DreamerV3', slug: 'algorithms/dreamerv3' },
								{ label: 'Dreamer 4', slug: 'algorithms/dreamer4' },
							],
						},
						{
							label: '6 · Planning (zero-series)',
							collapsed: true,
							items: [
								{ label: 'AlphaZero', slug: 'algorithms/alphazero' },
								{ label: 'MuZero', slug: 'algorithms/muzero' },
								{ label: 'EfficientZero V2', slug: 'algorithms/efficient-zero-v2' },
							],
						},
						{ label: 'Writing your own', slug: 'algorithms/custom' },
					],
				},
				{
					label: 'Neural networks',
					items: [
						{ label: 'Overview', slug: 'nn' },
						{ label: 'Modules and Params', slug: 'nn/modules' },
						{ label: 'Primitives', slug: 'nn/primitives' },
						{ label: 'Combinators and models', slug: 'nn/combinators' },
						{ label: 'Optimizers, losses, init', slug: 'nn/optimizers' },
						{ label: 'Training', slug: 'nn/training' },
					],
				},
				{
					label: 'Physics',
					items: [
						{ label: '3D engine', slug: 'physics/physics3d' },
						{ label: '2D engine', slug: 'physics/physics2d' },
						{ label: 'Validation', slug: 'physics/validation' },
					],
				},
				{
					label: 'Rendering',
					items: [{ label: 'Rendering', slug: 'rendering' }],
				},
				{
					label: 'Tooling',
					items: [
						{ label: 'Logging and metrics', slug: 'tooling/logging' },
						{ label: 'Training monitor', slug: 'tooling/monitor' },
					],
				},
				{
					label: 'Project',
					items: [
						{ label: 'Testing', slug: 'project/testing' },
						{ label: 'Contributing', slug: 'project/contributing' },
					],
				},
			],
		}),
	],
});

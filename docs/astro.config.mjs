import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://tyeoh9.github.io',
  base: '/llm-validation-framework',
  integrations: [
    starlight({
      title: 'validate-llm',
      description: 'Composable validation guardrails for your LLM pipelines.',
      logo: {
        light: './src/assets/lockup.svg',
        dark: './src/assets/lockup-dark.svg',
        replacesTitle: true,
      },
      favicon: '/favicon.svg',
      customCss: ['./src/styles/custom.css'],
      head: [
        {
          tag: 'script',
          content: `
            document.addEventListener('DOMContentLoaded', function () {
              document.querySelectorAll('.sl-markdown-content :is(h1,h2,h3,h4)').forEach(function (h) {
                if (!h.id) return;
                var a = document.createElement('a');
                a.href = '#' + h.id;
                a.className = 'anchor-link';
                a.setAttribute('aria-hidden', 'true');
                a.textContent = '#';
                h.appendChild(a);
              });
            });
          `,
        },
      ],
      social: {
        github: 'https://github.com/tyeoh9/llm-validation-framework',
      },
      sidebar: [
        { label: 'Getting Started', slug: 'getting-started' },
        {
          label: 'Core',
          items: [
            { label: 'ValidationFramework', slug: 'core/validation-framework' },
            { label: 'Pipe', slug: 'core/pipe' },
            { label: 'LLMProvider', slug: 'core/llm-provider' },
          ],
        },
        {
          label: 'Agents',
          items: [
            { label: 'ToxicityAgent', slug: 'agents/toxicity' },
            { label: 'PrivacyAgent', slug: 'agents/privacy' },
            { label: 'AccuracyAgent', slug: 'agents/accuracy' },
            { label: 'RelevancyAgent', slug: 'agents/relevancy' },
            { label: 'BiasAgent', slug: 'agents/bias' },
          ],
        },
        {
          label: 'Guides',
          items: [{ label: 'RAG Integration', slug: 'guides/rag-integration' }],
        },
        {
          label: 'Concepts',
          items: [
            { label: 'GEval: LLM-as-a-Judge', slug: 'concepts/geval' },
            { label: 'Toxicity: 3-Layer Detection', slug: 'concepts/toxicity-layers' },
            { label: 'Privacy: Pattern Detection', slug: 'concepts/privacy-patterns' },
          ],
        },
      ],
    }),
  ],
});

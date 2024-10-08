// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
import { themes } from "prism-react-renderer";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Developers | Wellcome Collection",
  tagline: "Make something new with our collections",
  url: "https://developers.wellcomecollection.org",
  baseUrl: "/",
  favicon: "/icons/favicon.ico",
  trailingSlash: false,
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",
  organizationName: "wellcomecollection",
  projectName: "developers.wellcomecollection.org",
  deploymentBranch: "gh-pages",

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          remarkPlugins: [require("mdx-mermaid")],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
    [
      "redocusaurus",
      {
        specs: [
          {
            route: "api/catalogue",
            spec: "reference/catalogue.yaml",
            layout: { title: "Catalogue API" },
          },
          {
            route: "api/iiif",
            spec: "reference/iiif.yaml",
            layout: { title: "IIIF APIs" },
          },
          {
            route: "api/content",
            spec: "reference/content.yaml",
            layout: { title: "Content API" },
          },
          // This is currently disabled until the documentation can be improved
          {
            route: "api/text",
            spec: "reference/text.yaml",
            layout: { title: "Text API" },
          },
        ],
        theme: {
          primaryColor: "#007868",
        },
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: "icons/android-192x192.png",
      navbar: {
        title: "Developers",
        hideOnScroll: true,
        logo: {
          alt: "Wellcome Collection",
          src: "images/wellcome-collection-black.svg",
          srcDark: "images/wellcome-collection-white.svg",
        },
        items: [
          {
            type: "dropdown",
            to: "docs/api",
            position: "left",
            label: "API Reference",
            items: [
              {
                to: "api/catalogue",
                label: "Catalogue",
              },
              {
                to: "api/iiif",
                label: "IIIF",
              },
              {
                to: "api/content",
                label: "Content",
              },
              // Hidden pending improved docs
              // {
              //   to: "api/text",
              //   label: "Text",
              // },
            ],
          },
          {
            href: "https://github.com/wellcomecollection",
            position: "right",
            label: "GitHub",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Documentation",
            items: [
              {
                label: "API Reference",
                to: "docs/api",
              },
            ],
          },
          {
            title: "Get Involved",
            items: [
              {
                label: "Stacks",
                href: "https://stacks.wellcomecollection.org",
              },
              {
                label: "GitHub",
                href: "https://github.com/wellcomecollection",
              },
            ],
          },
          {
            title: "Contact Us",
            items: [
              {
                label: "Twitter",
                href: "https://twitter.com/ExploreWellcome",
              },
              {
                label: "Facebook",
                href: "https://www.facebook.com/wellcomecollection",
              },
              {
                label: "Email",
                href: "mailto:digital@wellcomecollection.org",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "Jobs",
                href: "https://wellcome.org/jobs",
              },
              {
                label: "Privacy",
                href: "https://wellcome.org/who-we-are/privacy-and-terms",
              },
              {
                label: "Wellcome Collection",
                href: "https://wellcomecollection.org",
              },
            ],
          },
        ],
        copyright: `Except where otherwise noted, content on this site is licensed under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International Licence</a>.`,
      },
      prism: {
        theme: themes.vsLight,
        darkTheme: themes.vsDark,
      },
    }),

  plugins: [],
};

module.exports = config;

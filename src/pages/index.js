import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import HomepageFeatures from "@site/src/components/HomepageFeatures";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import clsx from "clsx";
import styles from "./index.module.css";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
function HomepageHeader({ description }) {
    const { siteConfig } = useDocusaurusContext();
    return (_jsx("header", { className: "hero shadow--lw", children: _jsxs("div", { className: clsx("container", styles.heroContainer), children: [_jsx("h1", { className: clsx("hero__title", styles.heroTitle), children: siteConfig.tagline }), _jsx("p", { className: clsx("hero__subtitle", styles.heroSubtitle), children: description }), _jsxs("div", { children: [_jsx(Link, { className: clsx("button button--primary button--outline button--lg", styles.button), to: "docs/api", children: "API Reference" }), " ", _jsx(Link, { className: clsx("button button--primary button--outline button--lg", styles.button), to: "docs/examples", children: "Examples" })] })] }) }));
}
export default function Home() {
    const siteDescription = "Wellcome Collection is a free museum and library that aims to challenge how we all think and feel about health. Find out how you can use open APIs to access our collections.";
    return (_jsxs(Layout, { description: siteDescription, children: [_jsx(HomepageHeader, { description: siteDescription }), _jsx("main", { children: _jsx(HomepageFeatures, {}) })] }));
}

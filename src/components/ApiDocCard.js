import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './ApiDocCard.module.css';
export default function ApiDocCard({ href, title, description }) {
    return (_jsxs(Link, { href: href, className: clsx('card margin-bottom--lg padding--lg', styles.cardContainer, styles.cardContainerLink), children: [_jsxs("h2", { className: clsx('text--truncate', styles.cardTitle), title: title, children: ["\uD83D\uDCC4\uFE0F ", title] }), _jsx("div", { className: clsx('text--truncate', styles.cardDescription), title: description, children: description })] }));
}

import { Fragment as _Fragment, jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import useBaseUrl from '@docusaurus/useBaseUrl';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';
const features = [
    {
        title: 'Catalogue API',
        image: 'images/index/catalogue.svg',
        link: '/docs/catalogue',
        description: (_jsx(_Fragment, { children: "Search our collections for visual culture, books, journals, archives, manuscripts and objects." })),
    },
    {
        title: 'IIIF APIs',
        image: 'images/index/iiif.svg',
        link: '/docs/iiif',
        description: (_jsx(_Fragment, { children: "Access digitised items using standard International Image Interoperability Framework (IIIF) APIs." })),
    }
];
function Feature({ title, image, description, link }) {
    return (_jsxs("a", { className: clsx('col'), href: link, children: [_jsx("div", { className: "text--center", children: _jsx("img", { className: styles.featureSvg, alt: title, src: useBaseUrl(image) }) }), _jsxs("div", { className: "text--center margin-vert--md padding-horiz--md", children: [_jsx("h3", { children: title }), _jsx("p", { children: description })] })] }));
}
export default function HomepageFeatures() {
    return (_jsx("section", { className: styles.features, children: _jsxs("div", { className: "container", children: [_jsx("div", { className: "text--center padding-vert--lg", children: _jsx("h2", { children: "We provide the following open APIs for accessing our collections" }) }), _jsx("div", { className: "row padding-vert--lg", children: features.map(props => (_jsx(Feature, { ...props }, props.link))) })] }) }));
}

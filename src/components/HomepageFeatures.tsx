import useBaseUrl from '@docusaurus/useBaseUrl';
import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

type FeatureItem = {
  title: string;
  image: string;
  description: JSX.Element;
  link: string;
};

const features: FeatureItem[] = [
  {
    title: 'Catalogue API',
    image: 'images/index/catalogue.svg',
    link: '/docs/catalogue',
    description: (
      <>
        Search our collections for visual culture, books, journals, archives, manuscripts and objects.
      </>
    ),
  },
  {
    title: 'IIIF APIs',
    image: 'images/index/iiif.svg',
    link: '/docs/iiif',
    description: (
      <>
        Access digitised items using standard International Image Interoperability Framework (IIIF) APIs.
      </>
    ),
  }
];

function Feature({title, image, description, link}: FeatureItem) {
  return (
    <a className={clsx('col')} href={link}>
      <div className="text--center">
        <img
          className={styles.featureSvg}
          alt={title}
          src={useBaseUrl(image)}
        />
      </div>
      <div className="text--center margin-vert--md padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </a>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
          <div className="text--center padding-vert--lg">
            <h2>We provide the following open APIs for accessing our collections</h2>
          </div>
        <div className="row padding-vert--lg">
          {features.map(props => (
            <Feature key={props.link} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

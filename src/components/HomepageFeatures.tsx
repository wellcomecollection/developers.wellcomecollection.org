import useBaseUrl from '@docusaurus/useBaseUrl';
import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

type FeatureItem = {
  title: string;
  image: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Catalogue API',
    image: 'images/index/catalogue.svg',
    description: (
      <>
        Search our collections for visual culture, books, journals, archives, manuscripts and objects.
      </>
    ),
  },
  {
    title: 'IIIF APIs',
    image: 'images/index/iiif.svg',
    description: (
      <>
        Access digitised items using standard International Image Interoperability Framework (IIIF) APIs.
      </>
    ),
  },
  {
    title: 'Text API',
    image: 'images/index/alto.svg',
    description: (
      <>
        Download the contents of digitised printed books, as raw text or structured ALTO XML.
      </>
    ),
  },
];

function Feature({title, image, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
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
    </div>
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
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

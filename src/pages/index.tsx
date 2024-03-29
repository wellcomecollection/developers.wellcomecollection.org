import HomepageFeatures from "@site/src/components/HomepageFeatures";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import React from "react";
import clsx from "clsx";
import styles from "./index.module.css";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

function HomepageHeader({ description }) {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className="hero shadow--lw">
      <div className={clsx("container", styles.heroContainer)}>
        <h1 className={clsx("hero__title", styles.heroTitle)}>
          {siteConfig.tagline}
        </h1>
        <p className={clsx("hero__subtitle", styles.heroSubtitle)}>
          {description}
        </p>
        <div>
          <Link
            className={clsx(
              "button button--primary button--outline button--lg",
              styles.button
            )}
            to="docs/api"
          >
            API Reference
          </Link>
         {" "}
          <Link
            className={clsx(
              "button button--primary button--outline button--lg",
              styles.button
            )}
            to="docs/examples"
          >
            Examples
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const siteDescription =
    "Wellcome Collection is a free museum and library that aims to challenge how we all think and feel about health. Find out how you can use open APIs to access our collections.";
  return (
    <Layout description={siteDescription}>
      <HomepageHeader description={siteDescription} />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}

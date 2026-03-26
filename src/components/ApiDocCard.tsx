import type { ReactNode } from 'react';
import Link from '@docusaurus/Link';
import clsx from 'clsx';
import styles from './ApiDocCard.module.css';

type CardOptions = {
  href: string;
  title: string;
  description: string;
}

export default function ApiDocCard({href, title, description}: CardOptions): ReactNode {
  return (
    <Link href={href} className={clsx(
      'card margin-bottom--lg padding--lg',
      styles.cardContainer,
      styles.cardContainerLink,
    )}>
      <h2 className={clsx('text--truncate', styles.cardTitle)} title={title}>
        📄️ {title}
      </h2>
      <div
        className={clsx('text--truncate', styles.cardDescription)}
        title={description}>
        {description}
      </div>
    </Link>
  );
}

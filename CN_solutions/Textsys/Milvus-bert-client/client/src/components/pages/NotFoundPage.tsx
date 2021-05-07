import React from 'react';
import { PageContent, PageWrapper } from '../../shared/Styles';

const NotFoundPage = () => {
  return (
    <PageWrapper>
      <PageContent>
        <div>
          Not found{' '}
          <span role="img" aria-label="sad face">
            😞
          </span>
        </div>
      </PageContent>
    </PageWrapper>
  );
};

export default NotFoundPage;

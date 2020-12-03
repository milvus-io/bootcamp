import React from 'react';
import styled from 'styled-components';

const LoadingWrapper = styled.div`
  margin: 150px auto;
`;

const Loading = () => {
  return (
    <LoadingWrapper>
      <svg
        width="48"
        height="48"
        viewBox="0 0 48 48"
        xmlns="http://www.w3.org/2000/svg"
        stroke="#323f50"
      >
        <g fill="none" fillRule="evenodd">
          <g transform="translate(4 4)" strokeWidth="8">
            <circle strokeOpacity=".5" cx="18" cy="18" r="18" />
            <path d="M36 18c0-9.94-8.06-18-18-18">
              <animateTransform
                attributeName="transform"
                type="rotate"
                from="0 18 18"
                to="360 18 18"
                dur="1s"
                repeatCount="indefinite"
              />
            </path>
          </g>
        </g>
      </svg>
    </LoadingWrapper>
  );
};

export default Loading;

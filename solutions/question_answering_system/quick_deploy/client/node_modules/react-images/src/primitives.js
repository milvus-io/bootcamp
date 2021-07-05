// @flow
// @jsx glam
import React from 'react';
import glam from 'glam';

type Props = {
  css?: {},
  innerRef?: HTMLElement => void,
  tag: string,
  ...{},
};

export const Base = ({ css, innerRef, tag: Tag, ...props }: Props) => (
  <Tag
    ref={innerRef}
    css={{
      boxSizing: 'border-box',
      ...css,
    }}
    {...props}
  />
);

export const A = (props: {}) => <Base tag="a" {...props} />;
export const Button = (props: {}) => <Base tag="button" {...props} />;
export const Div = (props: {}) => <Base tag="div" {...props} />;
export const Img = (props: {}) => <Base tag="img" {...props} />;
export const Nav = (props: {}) => <Base tag="nav" {...props} />;
export const Span = (props: {}) => <Base tag="span" {...props} />;

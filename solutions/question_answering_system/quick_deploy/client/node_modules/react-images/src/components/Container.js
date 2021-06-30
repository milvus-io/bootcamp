// @flow
// @jsx glam
import React, { type ElementRef, type Node } from 'react';
import glam from 'glam';
import { Div } from '../primitives';
import { type PropsWithStyles } from '../types';
import { className } from '../utils';

type State = { isFullscreen: boolean, isModal: boolean };
type Props = State &
  PropsWithStyles & {
    children: Node,
    innerProps: { innerRef: ElementRef<*> },
  };

export const containerCSS = ({ isFullscreen }: State) => ({
  backgroundColor: isFullscreen ? 'black' : null,
  display: 'flex ',
  flexDirection: 'column',
  height: '100%',
});

const Container = (props: Props) => {
  const { children, getStyles, isFullscreen, isModal, innerProps } = props;
  return (
    <Div
      css={getStyles('container', props)}
      className={className('container', { isFullscreen, isModal })}
      {...innerProps}
    >
      {children}
    </Div>
  );
};

export default Container;

// @flow
// @jsx glam
import React from 'react';
import glam from 'glam';
import { Div } from '../../primitives';
import { type PropsWithStyles } from '../../types';
import { className } from '../../utils';
import { FocusOn } from 'react-focus-on';
import BaseClassNames from '../componentBaseClassNames';

// ==============================
// Blanket
// ==============================

type BlanketState = { isFullscreen: boolean };
type BlanketProps = BlanketState &
  PropsWithStyles & {
    innerProps: Object, // TODO
  };

export const blanketCSS = ({ isFullscreen }: BlanketState) => ({
  backgroundColor: isFullscreen ? 'black' : 'rgba(0, 0, 0, 0.8)',
  bottom: 0,
  left: 0,
  position: 'fixed',
  right: 0,
  top: 0,
  zIndex: 1199,
});

export const Blanket = (props: BlanketProps) => {
  const { getStyles, innerProps, isFullscreen } = props;
  return (
    <Div
      css={getStyles('blanket', props)}
      className={className('blanket', { isFullscreen })}
      {...innerProps}
    />
  );
};

// ==============================
// Positioner
// ==============================

type PositionerState = { isFullscreen: boolean };
type PositionerProps = PositionerState &
  PropsWithStyles & {
    children: Node,
    innerProps: Object, // TODO
  };

export const positionerCSS = () => ({
  alignItems: 'center',
  bottom: 0,
  display: 'flex ',
  justifyContent: 'center',
  left: 0,
  position: 'fixed',
  right: 0,
  top: 0,
  zIndex: 1200,
});

export const Positioner = (props: PositionerProps) => {
  const { children, getStyles, innerProps, isFullscreen } = props;
  return (
    <Div
      css={getStyles(BaseClassNames.Positioner, props)}
      className={className(BaseClassNames.Positioner, { isFullscreen })}
      {...innerProps}
    >
      {children}
    </Div>
  );
};

// ==============================
// Dialog
// ==============================

type DialogState = { isFullscreen: boolean };
type DialogProps = DialogState &
  PropsWithStyles & {
    children: Node,
    innerProps: Object,
    removeFocusOn: Boolean,
  };

export const dialogCSS = () => ({
  width: '100%',
});

export const Dialog = (props: DialogProps) => {
  const {
    children,
    getStyles,
    innerProps,
    isFullscreen,
    removeFocusOn,
  } = props;

  return removeFocusOn ? (
    <Div
      css={getStyles('dialog', props)}
      className={className('dialog', { isFullscreen })}
      {...innerProps}
    >
      {children}
    </Div>
  ) : (
    <FocusOn>
      <Div
        css={getStyles('dialog', props)}
        className={className('dialog', { isFullscreen })}
        {...innerProps}
      >
        {children}
      </Div>
    </FocusOn>
  );
};

// @flow
// @jsx glam
import React, { type Node } from 'react';
import glam from 'glam';

import { Button, Nav } from '../primitives';
import { type PropsWithStyles } from '../types';
import { className, isTouch } from '../utils';
import { ChevronLeft, ChevronRight } from './svg';

// ==============================
// Navigation
// ==============================

type NavState = { interactionIsIdle: boolean };
type NavProps = NavState &
  PropsWithStyles & {
    children: Node,
    isFullscreen: boolean,
    isModal: boolean,
  };

export const navigationCSS = ({ interactionIsIdle }: NavState) => ({
  display: 'flex ',
  alignItems: 'center',
  justifyContent: 'space-between',
  opacity: interactionIsIdle ? 0 : 1,
  transition: 'opacity 300ms',
  '& *:focus': {
    outline: '1.5px solid orange',
  },
});

export const Navigation = (props: NavProps) => {
  const {
    children,
    getStyles,
    isFullscreen,
    isModal,
    showNavigationOnTouchDevice,
  } = props;
  return !isTouch() || (isTouch() && showNavigationOnTouchDevice) ? (
    <Nav
      css={getStyles('navigation', props)}
      className={className('navigation', { isFullscreen, isModal })}
    >
      {children}
    </Nav>
  ) : null;
};

// ==============================
// Nav Item
// ==============================

const BUTTON_SIZE = 50;

type ItemState = { align: 'left' | 'right' };
type ItemProps = ItemState &
  PropsWithStyles & {
    children: Node,
    innerProps: {
      onClick: (any) => void,
      title: string,
    },
  };

export const navigationItemCSS = ({ align }: ItemState) => ({
  alignItems: 'center',
  background: 'rgba(255, 255, 255, 0.2)',
  border: 0,
  borderRadius: '50%',
  color: 'white',
  cursor: 'pointer',
  display: 'flex ',
  fontSize: 'inherit',
  height: BUTTON_SIZE,
  justifyContent: 'center',
  marginTop: -(BUTTON_SIZE / 2),
  outline: 0,
  position: 'absolute',
  top: '50%',
  transition: 'background-color 200ms',
  width: BUTTON_SIZE,
  [align]: 20, // 'left' | 'right'

  '&:hover': {
    background: 'rgba(255, 255, 255, 0.3)',
  },
  '&:active': {
    background: 'rgba(255, 255, 255, 0.2)',
  },
});

export const navigationPrevCSS = navigationItemCSS;
export const NavigationPrev = (props: ItemProps) => {
  const { children = <ChevronLeft size={48} />, getStyles, innerProps } = props;

  return (
    <Button
      type="button"
      css={getStyles('navigationPrev', props)}
      {...innerProps}
    >
      {children}
    </Button>
  );
};

export const navigationNextCSS = navigationItemCSS;
export const NavigationNext = (props: ItemProps) => {
  const {
    children = <ChevronRight size={48} />,
    getStyles,
    innerProps,
  } = props;

  return (
    <Button
      type="button"
      css={getStyles('navigationNext', props)}
      {...innerProps}
    >
      {children}
    </Button>
  );
};

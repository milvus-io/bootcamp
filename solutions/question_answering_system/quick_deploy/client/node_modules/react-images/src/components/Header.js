// @flow
// @jsx glam
import React, { type Node } from 'react';
import glam from 'glam';

import { Button, Div } from '../primitives';
import { className } from '../utils';
import type { PropsWithStyles } from '../types';
import { Close, FullscreenEnter, FullscreenExit } from './svg';
import componentBaseClassNames from './componentBaseClassNames';

type State = { interactionIsIdle: boolean };
type Props = PropsWithStyles &
  State & {
    components: Object,
    getCloseLabel: Function,
    getFullscreenLabel: Function,
    innerProps: Object,
    isModal: boolean,
    modalProps: Object,
  };

export const headerCSS = ({ interactionIsIdle }: State) => ({
  alignItems: 'center',
  display: 'flex ',
  flex: '0 0 auto',
  justifyContent: 'space-between',
  opacity: interactionIsIdle ? 0 : 1,
  padding: 10,
  paddingBottom: 20,
  position: 'absolute',
  transform: `translateY(${interactionIsIdle ? -10 : 0}px)`,
  transition: 'opacity 300ms, transform 300ms',
  top: 0,
  left: 0,
  right: 0,
  zIndex: 1,
  '& *:focus': {
    outline: '1.5px solid orange',
  },
});

const headerBaseClassName = componentBaseClassNames.Header;

const Header = (props: Props) => {
  const {
    components,
    getStyles,
    getCloseLabel,
    getFullscreenLabel,
    innerProps,
    isModal,
    modalProps,
  } = props;

  if (!isModal) return null;

  const {
    allowFullscreen,
    isFullscreen,
    onClose,
    toggleFullscreen,
  } = modalProps;
  const FsIcon = isFullscreen ? FullscreenExit : FullscreenEnter;
  const { CloseButton, FullscreenButton } = components;
  const state = { isFullscreen, isModal };

  return (
    <Div
      css={getStyles(headerBaseClassName, props)}
      className={className(headerBaseClassName, state)}
      // TODO glam prefixer fails on gradients
      // https://github.com/threepointone/glam/issues/35
      style={{
        background: 'linear-gradient(rgba(0,0,0,0.33), rgba(0,0,0,0))',
      }}
      {...innerProps}
    >
      <span />
      <span>
        {allowFullscreen ? (
          <FullscreenButton
            getStyles={getStyles}
            innerProps={{
              onClick: toggleFullscreen,
              title: getFullscreenLabel(state),
            }}
          >
            <FsIcon size={32} />
          </FullscreenButton>
        ) : null}
        <CloseButton
          getStyles={getStyles}
          innerProps={{
            onClick: onClose,
            title: getCloseLabel(state),
          }}
        >
          <Close size={32} />
        </CloseButton>
      </span>
    </Div>
  );
};

// ==============================
// Header Buttons
// ==============================

type ButtonProps = Props & { children: Node };

export const headerButtonCSS = () => ({
  alignItems: 'center',
  background: 0,
  border: 0,
  color: 'rgba(255, 255, 255, 0.75)',
  cursor: 'pointer',
  display: 'inline-flex ',
  height: 44,
  justifyContent: 'center',
  outline: 0,
  padding: 0,
  position: 'relative',
  width: 44,

  '&:hover': {
    color: 'white',
  },
});

export const headerFullscreenCSS = headerButtonCSS;
export const HeaderFullscreen = (props: ButtonProps) => {
  const { children, getStyles, innerProps } = props;

  return (
    <Button
      css={getStyles('headerFullscreen', props)}
      className={className(['header_button', 'header_button--fullscreen'])}
      type="button"
      {...innerProps}
    >
      {children}
    </Button>
  );
};

export const headerCloseCSS = headerButtonCSS;
export const HeaderClose = (props: ButtonProps) => {
  const { children, getStyles, innerProps } = props;

  return (
    <Button
      css={getStyles('headerClose', props)}
      className={className(['header_button', 'header_button--close'])}
      type="button"
      {...innerProps}
    >
      {children}
    </Button>
  );
};

export default Header;

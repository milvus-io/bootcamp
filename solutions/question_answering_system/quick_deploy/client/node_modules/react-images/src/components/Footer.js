// @flow
// @jsx glam
import glam from 'glam';
import React from 'react';

import { smallDevice } from './css-helpers';
import { Div, Span } from '../primitives';
import type { PropsWithStyles, ViewType } from '../types';
import { className } from '../utils';
import componentBaseClassNames from './componentBaseClassNames';

import ParseHtml from 'html-react-parser';

type State = { isModal: boolean, interactionIsIdle: boolean };
type Props = State &
  PropsWithStyles & {
    components: Object,
    currentView: ViewType,
    innerProps: any,
    isFullscreen: boolean,
    isModal: boolean,
    modalProps: any,
    interactionIsIdle: any,
  };

export const footerCSS = ({ isModal, interactionIsIdle }: State) => ({
  alignItems: 'top',
  bottom: isModal ? 0 : null,
  color: isModal ? 'rgba(255, 255, 255, 0.9)' : '#666',
  display: 'flex ',
  flex: '0 0 auto',
  fontSize: 13,
  justifyContent: 'space-between',
  left: isModal ? 0 : null,
  opacity: interactionIsIdle && isModal ? 0 : 1,
  padding: isModal ? '30px 20px 20px' : '10px 0',
  position: isModal ? 'absolute' : null,
  right: isModal ? 0 : null,
  transform: isModal ? `translateY(${interactionIsIdle ? 10 : 0}px)` : null,
  transition: 'opacity 300ms, transform 300ms',
  zIndex: isModal ? 1 : null,

  '& *:focus': {
    outline: '1.5px solid orange',
  },

  [smallDevice]: {
    padding: isModal ? '20px 15px 15px' : '5px 0',
  },
});

const footerBaseClassName = componentBaseClassNames.Footer;

const Footer = (props: Props) => {
  const { components, getStyles, innerProps, isFullscreen, isModal } = props;

  const style = isModal
    ? { background: 'linear-gradient(rgba(0,0,0,0), rgba(0,0,0,0.33))' }
    : null;

  const state = { isFullscreen, isModal };
  const cn = {
    container: className(footerBaseClassName, state),
    caption: className('footer__caption', state),
    count: className('footer__count', state),
  };
  const css = {
    container: getStyles(footerBaseClassName, props),
    caption: getStyles('footerCaption', props),
    count: getStyles('footerCount', props),
  };
  const { Caption, Count } = components;

  return (
    <Div
      css={css.container}
      className={cn.container}
      // TODO glam prefixer fails on gradients
      // https://github.com/threepointone/glam/issues/35
      style={style}
      {...innerProps}
    >
      <Caption {...props} />
      <Count {...props} />
    </Div>
  );
};

// ==============================
// Inner Elements
// ==============================

export const footerCaptionCSS = () => ({});

export const FooterCaption = (props: ViewType) => {
  const { currentView, getStyles, isFullscreen, isModal } = props;
  const { caption } = currentView;
  const state = { isFullscreen, isModal };

  return (
    <Span
      css={getStyles('footerCaption', props)}
      className={className('footer__caption', state)}
    >
      {/*
       * Render a string caption using ParseHtml in case we were passed a
       * string with HTML. All other types are rendered by JSX by default.
       */}
      {typeof caption === 'string'
        ? ParseHtml(`<span>${caption}</span>`)
        : caption}
    </Span>
  );
};

export const footerCountCSS = () => ({ flexShrink: 0, marginLeft: '1em' });

export const FooterCount = (props: ViewType) => {
  const { currentIndex, getStyles, isFullscreen, isModal, views } = props;
  const state = { isFullscreen, isModal };
  const activeView = currentIndex + 1;
  const totalViews = views.length;

  if (!activeView || !totalViews) return null;

  return (
    <Span
      css={getStyles('footerCount', props)}
      className={className('footer__count', state)}
    >
      {activeView} of {totalViews}
    </Span>
  );
};

export default Footer;

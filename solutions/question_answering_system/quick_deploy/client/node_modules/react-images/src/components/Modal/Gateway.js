// @flow
import React, { Children, Component } from 'react';
import { createPortal } from 'react-dom';
import { TransitionGroup } from 'react-transition-group';

import { type ModalType } from './Modal';

const FirstChild = ({ children }) => Children.toArray(children)[0] || null;

export default class ModalGateway extends Component<{
  children: ModalType,
  target: HTMLElement,
}> {
  static defaultProps = {
    target: (typeof(window) !== 'undefined') ? document.body : null,
  };
  render() {
    if (typeof(window) === 'undefined') return null;
    const { target, children } = this.props;
    return createPortal(
      <TransitionGroup component={FirstChild} children={children} />,
      target,
    );
  }
}

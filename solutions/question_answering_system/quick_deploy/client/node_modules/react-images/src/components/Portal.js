import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { CSSTransition, TransitionGroup } from 'react-transition-group';
import { render, unmountComponentAtNode } from 'react-dom';
import PassContext from './PassContext';

export default class Portal extends Component {
  constructor() {
    super();
    this.portalElement = null;
  }
  componentDidMount() {
    const p = document.createElement('div');
    document.body.appendChild(p);
    this.portalElement = p;
    this.componentDidUpdate();
  }
  componentDidUpdate() {
    // Animate fade on mount/unmount
    const duration = 200;
    const styles = `
				.fade-enter { opacity: 0.01; }
				.fade-enter.fade-enter-active { opacity: 1; transition: opacity ${duration}ms; }
				.fade-leave { opacity: 1; }
				.fade-leave.fade-leave-active { opacity: 0.01; transition: opacity ${duration}ms; }
		`;

    render(
      <PassContext context={this.context}>
        <div>
          <style>{styles}</style>
          <TransitionGroup {...this.props}>
            <CSSTransition
              timeout={{ enter: duration, exit: duration }}
              classNames="fade"
            >
              {this.props.children}
            </CSSTransition>
          </TransitionGroup>
        </div>
      </PassContext>,
      this.portalElement
    );
  }
  componentWillUnmount() {
    unmountComponentAtNode(this.portalElement);
    document.body.removeChild(this.portalElement);
  }
  render() {
    return null;
  }
}

Portal.contextTypes = {
  theme: PropTypes.object.isRequired,
};

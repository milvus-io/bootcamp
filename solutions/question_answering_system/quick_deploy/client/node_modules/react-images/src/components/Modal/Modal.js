// @flow
// @jsx glam
import React, { cloneElement, Component } from 'react';
import glam from 'glam';
import Fullscreen from 'react-full-screen';
import focusStore from 'a11y-focus-store';
import {
  defaultModalComponents,
  type ModalComponents,
} from '../defaultComponents';
import { Fade, SlideUp } from './Animation';
import { type CarouselType } from '../Carousel';
import { defaultModalStyles, type ModalStylesConfig } from '../../styles';
import { isTouch, className } from '../../utils';
import componentBaseClassNames from '../componentBaseClassNames';

type MouseOrKeyboardEvent = MouseEvent | KeyboardEvent;
export type CloseType = (event: MouseOrKeyboardEvent) => void;
export type ModalProps = {
  allowFullscreen: boolean,
  isFullscreen: boolean,
  onClose: CloseType,
  preventScroll: boolean,
  toggleFullscreen: (any) => void,
};

export type Props = {
  /* Enable/disable the ability to "fullscreen" the dialog */
  allowFullscreen: boolean,
  /* Carousel only supported */
  children: CarouselType,
  /* Enable/disable calling `onClose` when the backdrop is clicked */
  closeOnBackdropClick: boolean,
  /* Enable/disable calling `onClose` when the `esc` key is pressed */
  closeOnEsc: boolean,
  /* Replace any of the modal components */
  components?: ModalComponents,
  /*
    Show the component; triggers the enter or exit states
    NOTE: provided by TransitionGroup, NOT supplied by the user
  */
  in: boolean,
  /* Function called to request close of the modal */
  onClose: CloseType,
  /* Prevent scroll */
  preventScroll: Boolean,
  /* Style modifier methods */
  styles: ModalStylesConfig,
};
type State = { isFullscreen: boolean };
const defaultProps = {
  allowFullscreen: !isTouch(),
  closeOnBackdropClick: true,
  closeOnEsc: true,
  preventScroll: true,
  styles: {},
};

/** Classes that when clicked on, close the backdrop */
const backdropClassNames = new Set(
  [
    componentBaseClassNames.View,
    componentBaseClassNames.Header,
    componentBaseClassNames.Footer,
    componentBaseClassNames.Track,
    componentBaseClassNames.Positioner,
  ].map(className)
);

class Modal extends Component<Props, State> {
  commonProps: any; // TODO
  components: ModalComponents;

  static defaultProps = defaultProps;

  constructor(props: Props) {
    super(props);

    this.cacheComponents(props.components);

    this.state = { isFullscreen: false, isClosing: false };
  }

  componentDidUpdate(prevProps: Props) {
    if (prevProps.components !== this.props.components) {
      this.cacheComponents(prevProps.components);
    }
  }

  // emulate `componentDidMount` & `componentWillUnmount`
  // called on complete of enter & exit transitions respectively
  modalDidMount = () => {
    document.addEventListener('keyup', this.handleKeyUp);
    focusStore.storeFocus();
  };
  modalWillUnmount = () => {
    document.removeEventListener('keyup', this.handleKeyUp);
    focusStore.restoreFocus();
    this.setState({ isClosing: false });
  };

  cacheComponents = (comps?: ModalComponents) => {
    this.components = defaultModalComponents(comps);
  };
  handleFullscreenChange = (isFullscreen: boolean) => {
    this.setState({ isFullscreen });
  };
  handleKeyUp = (event: KeyboardEvent) => {
    const { allowFullscreen, closeOnEsc } = this.props;
    const { isFullscreen } = this.state;
    const allowClose = event.key === 'Escape' && closeOnEsc && !isFullscreen;

    // toggle fullscreen
    if (allowFullscreen && event.key === 'f') {
      this.toggleFullscreen();
    }

    // close on escape when not fullscreen
    if (allowClose) this.handleClose(event);
  };
  handleBackdropClick = (event: MouseEvent) => {
    let hasBackdropClassName = false;
    for (const targetClass of event.target.classList) {
      if (backdropClassNames.has(targetClass)) {
        hasBackdropClassName = true;
      }
    }

    if (!hasBackdropClassName || !this.props.closeOnBackdropClick) {
      return;
    }

    this.handleClose(event);
  };
  toggleFullscreen = () => {
    this.setState((state) => ({ isFullscreen: !state.isFullscreen }));
  };
  handleClose = (event: MouseOrKeyboardEvent) => {
    const { onClose } = this.props;
    const { isFullscreen } = this.state;

    this.setState({ isClosing: true });

    // force exit fullscreen mode on close
    if (isFullscreen) {
      this.toggleFullscreen();
    }

    // call the consumer's onClose func
    onClose(event);
  };

  getStyles = (key: string, props: {}): {} => {
    const base = defaultModalStyles[key](props);
    base.boxSizing = 'border-box';
    const custom = this.props.styles[key];
    return custom ? custom(base, props) : base;
  };
  getCommonProps() {
    const { isFullscreen } = this.state;

    return {
      getStyles: this.getStyles,
      isFullscreen,
      modalProps: this.props,
    };
  }
  render() {
    const { Blanket, Positioner, Dialog } = this.components;
    const { allowFullscreen, children } = this.props;
    const { isFullscreen } = this.state;
    const commonProps = (this.commonProps = this.getCommonProps());

    // $FlowFixMe
    const transitionIn = this.props.in;

    // forward props to modal for use in internal components
    const modalProps: ModalProps = {
      allowFullscreen,
      isFullscreen,
      onClose: this.handleClose,
      preventScroll: this.preventScroll,
      toggleFullscreen: this.toggleFullscreen,
    };

    // augment user carousel with modal props
    // $FlowFixMe
    const carouselComponent: CarouselType = cloneElement(children, {
      isModal: true,
      modalProps,
    });

    return (
      <Fullscreen enabled={isFullscreen} onChange={this.handleFullscreenChange}>
        <Fade {...commonProps} component={Blanket} in={transitionIn} />
        <SlideUp
          {...commonProps}
          component={Positioner}
          in={transitionIn}
          innerProps={{
            onClick: this.state.isClosing ? null : this.handleBackdropClick,
          }}
          onEntered={this.modalDidMount}
          onExited={this.modalWillUnmount}
        >
          <Dialog removeFocusOn={this.state.isClosing} {...commonProps}>
            {carouselComponent}
          </Dialog>
        </SlideUp>
      </Fullscreen>
    );
  }
}
export default Modal;
export type ModalType = typeof Modal;

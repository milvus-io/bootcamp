import _extends from "@babel/runtime/helpers/esm/extends";
import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";
import _assertThisInitialized from "@babel/runtime/helpers/esm/assertThisInitialized";
import _inheritsLoose from "@babel/runtime/helpers/esm/inheritsLoose";

/* eslint-disable react/prop-types */
import activeElement from 'dom-helpers/activeElement';
import contains from 'dom-helpers/contains';
import canUseDOM from 'dom-helpers/canUseDOM';
import listen from 'dom-helpers/listen';
import PropTypes from 'prop-types';
import React from 'react';
import ReactDOM from 'react-dom';
import ModalManager from './ModalManager';
import ownerDocument from './utils/ownerDocument';
import useWaitForDOMRef from './utils/useWaitForDOMRef';

function omitProps(props, propTypes) {
  var keys = Object.keys(props);
  var newProps = {};
  keys.forEach(function (prop) {
    if (!Object.prototype.hasOwnProperty.call(propTypes, prop)) {
      newProps[prop] = props[prop];
    }
  });
  return newProps;
}

var manager;
/**
 * Love them or hate them, `<Modal />` provides a solid foundation for creating dialogs, lightboxes, or whatever else.
 * The Modal component renders its `children` node in front of a backdrop component.
 *
 * The Modal offers a few helpful features over using just a `<Portal/>` component and some styles:
 *
 * - Manages dialog stacking when one-at-a-time just isn't enough.
 * - Creates a backdrop, for disabling interaction below the modal.
 * - It properly manages focus; moving to the modal content, and keeping it there until the modal is closed.
 * - It disables scrolling of the page content while open.
 * - Adds the appropriate ARIA roles are automatically.
 * - Easily pluggable animations via a `<Transition/>` component.
 *
 * Note that, in the same way the backdrop element prevents users from clicking or interacting
 * with the page content underneath the Modal, Screen readers also need to be signaled to not to
 * interact with page content while the Modal is open. To do this, we use a common technique of applying
 * the `aria-hidden='true'` attribute to the non-Modal elements in the Modal `container`. This means that for
 * a Modal to be truly modal, it should have a `container` that is _outside_ your app's
 * React hierarchy (such as the default: document.body).
 */

var Modal =
/*#__PURE__*/
function (_React$Component) {
  _inheritsLoose(Modal, _React$Component);

  function Modal() {
    var _this;

    for (var _len = arguments.length, _args = new Array(_len), _key = 0; _key < _len; _key++) {
      _args[_key] = arguments[_key];
    }

    _this = _React$Component.call.apply(_React$Component, [this].concat(_args)) || this;
    _this.state = {
      exited: !_this.props.show
    };

    _this.onShow = function () {
      var _this$props = _this.props,
          container = _this$props.container,
          containerClassName = _this$props.containerClassName,
          onShow = _this$props.onShow;

      _this.getModalManager().add(_assertThisInitialized(_this), container, containerClassName);

      _this.removeKeydownListener = listen(document, 'keydown', _this.handleDocumentKeyDown);
      _this.removeFocusListener = listen(document, 'focus', // the timeout is necessary b/c this will run before the new modal is mounted
      // and so steals focus from it
      function () {
        return setTimeout(_this.enforceFocus);
      }, true);

      if (onShow) {
        onShow();
      } // autofocus after onShow, to not trigger a focus event for previous
      // modals before this one is shown.


      _this.autoFocus();
    };

    _this.onHide = function () {
      _this.getModalManager().remove(_assertThisInitialized(_this));

      _this.removeKeydownListener();

      _this.removeFocusListener();

      if (_this.props.restoreFocus) {
        _this.restoreLastFocus();
      }
    };

    _this.setDialogRef = function (ref) {
      _this.dialog = ref;
    };

    _this.setBackdropRef = function (ref) {
      _this.backdrop = ref && ReactDOM.findDOMNode(ref);
    };

    _this.handleHidden = function () {
      _this.setState({
        exited: true
      });

      _this.onHide();

      if (_this.props.onExited) {
        var _this$props2;

        (_this$props2 = _this.props).onExited.apply(_this$props2, arguments);
      }
    };

    _this.handleBackdropClick = function (e) {
      if (e.target !== e.currentTarget) {
        return;
      }

      if (_this.props.onBackdropClick) {
        _this.props.onBackdropClick(e);
      }

      if (_this.props.backdrop === true) {
        _this.props.onHide();
      }
    };

    _this.handleDocumentKeyDown = function (e) {
      if (_this.props.keyboard && e.keyCode === 27 && _this.isTopModal()) {
        if (_this.props.onEscapeKeyDown) {
          _this.props.onEscapeKeyDown(e);
        }

        _this.props.onHide();
      }
    };

    _this.enforceFocus = function () {
      if (!_this.props.enforceFocus || !_this._isMounted || !_this.isTopModal()) {
        return;
      }

      var currentActiveElement = activeElement(ownerDocument(_assertThisInitialized(_this)));

      if (_this.dialog && !contains(_this.dialog, currentActiveElement)) {
        _this.dialog.focus();
      }
    };

    _this.renderBackdrop = function () {
      var _this$props3 = _this.props,
          renderBackdrop = _this$props3.renderBackdrop,
          Transition = _this$props3.backdropTransition;
      var backdrop = renderBackdrop({
        ref: _this.setBackdropRef,
        onClick: _this.handleBackdropClick
      });

      if (Transition) {
        backdrop = React.createElement(Transition, {
          appear: true,
          "in": _this.props.show
        }, backdrop);
      }

      return backdrop;
    };

    return _this;
  }

  Modal.getDerivedStateFromProps = function getDerivedStateFromProps(nextProps) {
    if (nextProps.show) {
      return {
        exited: false
      };
    }

    if (!nextProps.transition) {
      // Otherwise let handleHidden take care of marking exited.
      return {
        exited: true
      };
    }

    return null;
  };

  var _proto = Modal.prototype;

  _proto.componentDidMount = function componentDidMount() {
    this._isMounted = true;

    if (this.props.show) {
      this.onShow();
    }
  };

  _proto.componentDidUpdate = function componentDidUpdate(prevProps) {
    var transition = this.props.transition;

    if (prevProps.show && !this.props.show && !transition) {
      // Otherwise handleHidden will call this.
      this.onHide();
    } else if (!prevProps.show && this.props.show) {
      this.onShow();
    }
  };

  _proto.componentWillUnmount = function componentWillUnmount() {
    var _this$props4 = this.props,
        show = _this$props4.show,
        transition = _this$props4.transition;
    this._isMounted = false;

    if (show || transition && !this.state.exited) {
      this.onHide();
    }
  };

  _proto.getSnapshotBeforeUpdate = function getSnapshotBeforeUpdate(prevProps) {
    if (canUseDOM && !prevProps.show && this.props.show) {
      this.lastFocus = activeElement();
    }

    return null;
  };

  _proto.getModalManager = function getModalManager() {
    if (this.props.manager) {
      return this.props.manager;
    }

    if (!manager) {
      manager = new ModalManager();
    }

    return manager;
  };

  _proto.restoreLastFocus = function restoreLastFocus() {
    // Support: <=IE11 doesn't support `focus()` on svg elements (RB: #917)
    if (this.lastFocus && this.lastFocus.focus) {
      this.lastFocus.focus(this.props.restoreFocusOptions);
      this.lastFocus = null;
    }
  };

  _proto.autoFocus = function autoFocus() {
    if (!this.props.autoFocus) return;
    var currentActiveElement = activeElement(ownerDocument(this));

    if (this.dialog && !contains(this.dialog, currentActiveElement)) {
      this.lastFocus = currentActiveElement;
      this.dialog.focus();
    }
  };

  _proto.isTopModal = function isTopModal() {
    return this.getModalManager().isTopModal(this);
  };

  _proto.render = function render() {
    var _this$props5 = this.props,
        show = _this$props5.show,
        container = _this$props5.container,
        children = _this$props5.children,
        renderDialog = _this$props5.renderDialog,
        _this$props5$role = _this$props5.role,
        role = _this$props5$role === void 0 ? 'dialog' : _this$props5$role,
        Transition = _this$props5.transition,
        backdrop = _this$props5.backdrop,
        className = _this$props5.className,
        style = _this$props5.style,
        onExit = _this$props5.onExit,
        onExiting = _this$props5.onExiting,
        onEnter = _this$props5.onEnter,
        onEntering = _this$props5.onEntering,
        onEntered = _this$props5.onEntered,
        props = _objectWithoutPropertiesLoose(_this$props5, ["show", "container", "children", "renderDialog", "role", "transition", "backdrop", "className", "style", "onExit", "onExiting", "onEnter", "onEntering", "onEntered"]);

    if (!(show || Transition && !this.state.exited)) {
      return null;
    }

    var dialogProps = _extends({
      role: role,
      ref: this.setDialogRef,
      // apparently only works on the dialog role element
      'aria-modal': role === 'dialog' ? true : undefined
    }, omitProps(props, Modal.propTypes), {
      style: style,
      className: className,
      tabIndex: '-1'
    });

    var dialog = renderDialog ? renderDialog(dialogProps) : React.createElement("div", dialogProps, React.cloneElement(children, {
      role: 'document'
    }));

    if (Transition) {
      dialog = React.createElement(Transition, {
        appear: true,
        unmountOnExit: true,
        "in": show,
        onExit: onExit,
        onExiting: onExiting,
        onExited: this.handleHidden,
        onEnter: onEnter,
        onEntering: onEntering,
        onEntered: onEntered
      }, dialog);
    }

    return ReactDOM.createPortal(React.createElement(React.Fragment, null, backdrop && this.renderBackdrop(), dialog), container);
  };

  return Modal;
}(React.Component); // dumb HOC for the sake react-docgen


Modal.propTypes = {
  /**
   * Set the visibility of the Modal
   */
  show: PropTypes.bool,

  /**
   * A DOM element, a `ref` to an element, or function that returns either. The Modal is appended to it's `container` element.
   *
   * For the sake of assistive technologies, the container should usually be the document body, so that the rest of the
   * page content can be placed behind a virtual backdrop as well as a visual one.
   */
  container: PropTypes.any,

  /**
   * A callback fired when the Modal is opening.
   */
  onShow: PropTypes.func,

  /**
   * A callback fired when either the backdrop is clicked, or the escape key is pressed.
   *
   * The `onHide` callback only signals intent from the Modal,
   * you must actually set the `show` prop to `false` for the Modal to close.
   */
  onHide: PropTypes.func,

  /**
   * Include a backdrop component.
   */
  backdrop: PropTypes.oneOfType([PropTypes.bool, PropTypes.oneOf(['static'])]),

  /**
   * A function that returns the dialog component. Useful for custom
   * rendering. **Note:** the component should make sure to apply the provided ref.
   *
   * ```js
   *  renderDialog={props => <MyDialog {...props} />}
   * ```
   */
  renderDialog: PropTypes.func,

  /**
   * A function that returns a backdrop component. Useful for custom
   * backdrop rendering.
   *
   * ```js
   *  renderBackdrop={props => <MyBackdrop {...props} />}
   * ```
   */
  renderBackdrop: PropTypes.func,

  /**
   * A callback fired when the escape key, if specified in `keyboard`, is pressed.
   */
  onEscapeKeyDown: PropTypes.func,

  /**
   * A callback fired when the backdrop, if specified, is clicked.
   */
  onBackdropClick: PropTypes.func,

  /**
   * A css class or set of classes applied to the modal container when the modal is open,
   * and removed when it is closed.
   */
  containerClassName: PropTypes.string,

  /**
   * Close the modal when escape key is pressed
   */
  keyboard: PropTypes.bool,

  /**
   * A `react-transition-group@2.0.0` `<Transition/>` component used
   * to control animations for the dialog component.
   */
  transition: PropTypes.elementType,

  /**
   * A `react-transition-group@2.0.0` `<Transition/>` component used
   * to control animations for the backdrop components.
   */
  backdropTransition: PropTypes.elementType,

  /**
   * When `true` The modal will automatically shift focus to itself when it opens, and
   * replace it to the last focused element when it closes. This also
   * works correctly with any Modal children that have the `autoFocus` prop.
   *
   * Generally this should never be set to `false` as it makes the Modal less
   * accessible to assistive technologies, like screen readers.
   */
  autoFocus: PropTypes.bool,

  /**
   * When `true` The modal will prevent focus from leaving the Modal while open.
   *
   * Generally this should never be set to `false` as it makes the Modal less
   * accessible to assistive technologies, like screen readers.
   */
  enforceFocus: PropTypes.bool,

  /**
   * When `true` The modal will restore focus to previously focused element once
   * modal is hidden
   */
  restoreFocus: PropTypes.bool,

  /**
   * Options passed to focus function when `restoreFocus` is set to `true`
   *
   * @link  https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/focus#Parameters
   */
  restoreFocusOptions: PropTypes.shape({
    preventScroll: PropTypes.bool
  }),

  /**
   * Callback fired before the Modal transitions in
   */
  onEnter: PropTypes.func,

  /**
   * Callback fired as the Modal begins to transition in
   */
  onEntering: PropTypes.func,

  /**
   * Callback fired after the Modal finishes transitioning in
   */
  onEntered: PropTypes.func,

  /**
   * Callback fired right before the Modal transitions out
   */
  onExit: PropTypes.func,

  /**
   * Callback fired as the Modal begins to transition out
   */
  onExiting: PropTypes.func,

  /**
   * Callback fired after the Modal finishes transitioning out
   */
  onExited: PropTypes.func,

  /**
   * A ModalManager instance used to track and manage the state of open
   * Modals. Useful when customizing how modals interact within a container
   */
  manager: PropTypes.object
};
Modal.defaultProps = {
  show: false,
  role: 'dialog',
  backdrop: true,
  keyboard: true,
  autoFocus: true,
  enforceFocus: true,
  restoreFocus: true,
  onHide: function onHide() {},
  renderBackdrop: function renderBackdrop(props) {
    return React.createElement("div", props);
  }
};

function forwardRef(Component) {
  // eslint-disable-next-line react/display-name
  var ModalWithContainer = React.forwardRef(function (props, ref) {
    var resolved = useWaitForDOMRef(props.container);
    return resolved ? React.createElement(Component, _extends({}, props, {
      ref: ref,
      container: resolved
    })) : null;
  });
  ModalWithContainer.Manager = ModalManager;
  ModalWithContainer._Inner = Component;
  return ModalWithContainer;
}

var ModalWithContainer = forwardRef(Modal);
ModalWithContainer.Manager = ModalManager;
export default ModalWithContainer;
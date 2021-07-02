"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _reactDom = _interopRequireDefault(require("react-dom"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _ownerDocument = _interopRequireDefault(require("../utils/ownerDocument"));

var _useForkRef = _interopRequireDefault(require("../utils/useForkRef"));

var _setRef = _interopRequireDefault(require("../utils/setRef"));

var _useEventCallback = _interopRequireDefault(require("../utils/useEventCallback"));

var _utils = require("@material-ui/utils");

function mapEventPropToEvent(eventProp) {
  return eventProp.substring(2).toLowerCase();
}
/**
 * Listen for click events that occur somewhere in the document, outside of the element itself.
 * For instance, if you need to hide a menu when people click anywhere else on your page.
 */


var ClickAwayListener = _react.default.forwardRef(function ClickAwayListener(props, ref) {
  var children = props.children,
      _props$mouseEvent = props.mouseEvent,
      mouseEvent = _props$mouseEvent === void 0 ? 'onClick' : _props$mouseEvent,
      _props$touchEvent = props.touchEvent,
      touchEvent = _props$touchEvent === void 0 ? 'onTouchEnd' : _props$touchEvent,
      onClickAway = props.onClickAway;

  var movedRef = _react.default.useRef(false);

  var nodeRef = _react.default.useRef(null);

  var mountedRef = _react.default.useRef(false);

  _react.default.useEffect(function () {
    mountedRef.current = true;
    return function () {
      mountedRef.current = false;
    };
  }, []);

  var handleNodeRef = (0, _useForkRef.default)(nodeRef, ref); // can be removed once we drop support for non ref forwarding class components

  var handleOwnRef = _react.default.useCallback(function (instance) {
    // #StrictMode ready
    (0, _setRef.default)(handleNodeRef, _reactDom.default.findDOMNode(instance));
  }, [handleNodeRef]);

  var handleRef = (0, _useForkRef.default)(children.ref, handleOwnRef);
  var handleClickAway = (0, _useEventCallback.default)(function (event) {
    // The handler doesn't take event.defaultPrevented into account:
    //
    // event.preventDefault() is meant to stop default behaviours like
    // clicking a checkbox to check it, hitting a button to submit a form,
    // and hitting left arrow to move the cursor in a text input etc.
    // Only special HTML elements have these default behaviors.
    // IE 11 support, which trigger the handleClickAway even after the unbind
    if (!mountedRef.current) {
      return;
    } // Do not act if user performed touchmove


    if (movedRef.current) {
      movedRef.current = false;
      return;
    } // The child might render null.


    if (!nodeRef.current) {
      return;
    } // Multi window support


    var doc = (0, _ownerDocument.default)(nodeRef.current);

    if (doc.documentElement && doc.documentElement.contains(event.target) && !nodeRef.current.contains(event.target)) {
      onClickAway(event);
    }
  });

  var handleTouchMove = _react.default.useCallback(function () {
    movedRef.current = true;
  }, []);

  _react.default.useEffect(function () {
    if (touchEvent !== false) {
      var mappedTouchEvent = mapEventPropToEvent(touchEvent);
      var doc = (0, _ownerDocument.default)(nodeRef.current);
      doc.addEventListener(mappedTouchEvent, handleClickAway);
      doc.addEventListener('touchmove', handleTouchMove);
      return function () {
        doc.removeEventListener(mappedTouchEvent, handleClickAway);
        doc.removeEventListener('touchmove', handleTouchMove);
      };
    }

    return undefined;
  }, [handleClickAway, handleTouchMove, touchEvent]);

  _react.default.useEffect(function () {
    if (mouseEvent !== false) {
      var mappedMouseEvent = mapEventPropToEvent(mouseEvent);
      var doc = (0, _ownerDocument.default)(nodeRef.current);
      doc.addEventListener(mappedMouseEvent, handleClickAway);
      return function () {
        doc.removeEventListener(mappedMouseEvent, handleClickAway);
      };
    }

    return undefined;
  }, [handleClickAway, mouseEvent]);

  return _react.default.createElement(_react.default.Fragment, null, _react.default.cloneElement(children, {
    ref: handleRef
  }));
});

process.env.NODE_ENV !== "production" ? ClickAwayListener.propTypes = {
  /**
   * The wrapped element.
   */
  children: _utils.elementAcceptingRef.isRequired,

  /**
   * The mouse event to listen to. You can disable the listener by providing `false`.
   */
  mouseEvent: _propTypes.default.oneOf(['onClick', 'onMouseDown', 'onMouseUp', false]),

  /**
   * Callback fired when a "click away" event is detected.
   */
  onClickAway: _propTypes.default.func.isRequired,

  /**
   * The touch event to listen to. You can disable the listener by providing `false`.
   */
  touchEvent: _propTypes.default.oneOf(['onTouchStart', 'onTouchEnd', false])
} : void 0;

if (process.env.NODE_ENV !== 'production') {
  // eslint-disable-next-line
  ClickAwayListener['propTypes' + ''] = (0, _utils.exactProp)(ClickAwayListener.propTypes);
}

var _default = ClickAwayListener;
exports.default = _default;
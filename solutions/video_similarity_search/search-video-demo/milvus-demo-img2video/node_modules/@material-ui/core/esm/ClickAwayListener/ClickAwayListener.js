import React from 'react';
import ReactDOM from 'react-dom';
import PropTypes from 'prop-types';
import ownerDocument from '../utils/ownerDocument';
import useForkRef from '../utils/useForkRef';
import setRef from '../utils/setRef';
import useEventCallback from '../utils/useEventCallback';
import { elementAcceptingRef, exactProp } from '@material-ui/utils';

function mapEventPropToEvent(eventProp) {
  return eventProp.substring(2).toLowerCase();
}
/**
 * Listen for click events that occur somewhere in the document, outside of the element itself.
 * For instance, if you need to hide a menu when people click anywhere else on your page.
 */


var ClickAwayListener = React.forwardRef(function ClickAwayListener(props, ref) {
  var children = props.children,
      _props$mouseEvent = props.mouseEvent,
      mouseEvent = _props$mouseEvent === void 0 ? 'onClick' : _props$mouseEvent,
      _props$touchEvent = props.touchEvent,
      touchEvent = _props$touchEvent === void 0 ? 'onTouchEnd' : _props$touchEvent,
      onClickAway = props.onClickAway;
  var movedRef = React.useRef(false);
  var nodeRef = React.useRef(null);
  var mountedRef = React.useRef(false);
  React.useEffect(function () {
    mountedRef.current = true;
    return function () {
      mountedRef.current = false;
    };
  }, []);
  var handleNodeRef = useForkRef(nodeRef, ref); // can be removed once we drop support for non ref forwarding class components

  var handleOwnRef = React.useCallback(function (instance) {
    // #StrictMode ready
    setRef(handleNodeRef, ReactDOM.findDOMNode(instance));
  }, [handleNodeRef]);
  var handleRef = useForkRef(children.ref, handleOwnRef);
  var handleClickAway = useEventCallback(function (event) {
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


    var doc = ownerDocument(nodeRef.current);

    if (doc.documentElement && doc.documentElement.contains(event.target) && !nodeRef.current.contains(event.target)) {
      onClickAway(event);
    }
  });
  var handleTouchMove = React.useCallback(function () {
    movedRef.current = true;
  }, []);
  React.useEffect(function () {
    if (touchEvent !== false) {
      var mappedTouchEvent = mapEventPropToEvent(touchEvent);
      var doc = ownerDocument(nodeRef.current);
      doc.addEventListener(mappedTouchEvent, handleClickAway);
      doc.addEventListener('touchmove', handleTouchMove);
      return function () {
        doc.removeEventListener(mappedTouchEvent, handleClickAway);
        doc.removeEventListener('touchmove', handleTouchMove);
      };
    }

    return undefined;
  }, [handleClickAway, handleTouchMove, touchEvent]);
  React.useEffect(function () {
    if (mouseEvent !== false) {
      var mappedMouseEvent = mapEventPropToEvent(mouseEvent);
      var doc = ownerDocument(nodeRef.current);
      doc.addEventListener(mappedMouseEvent, handleClickAway);
      return function () {
        doc.removeEventListener(mappedMouseEvent, handleClickAway);
      };
    }

    return undefined;
  }, [handleClickAway, mouseEvent]);
  return React.createElement(React.Fragment, null, React.cloneElement(children, {
    ref: handleRef
  }));
});
process.env.NODE_ENV !== "production" ? ClickAwayListener.propTypes = {
  /**
   * The wrapped element.
   */
  children: elementAcceptingRef.isRequired,

  /**
   * The mouse event to listen to. You can disable the listener by providing `false`.
   */
  mouseEvent: PropTypes.oneOf(['onClick', 'onMouseDown', 'onMouseUp', false]),

  /**
   * Callback fired when a "click away" event is detected.
   */
  onClickAway: PropTypes.func.isRequired,

  /**
   * The touch event to listen to. You can disable the listener by providing `false`.
   */
  touchEvent: PropTypes.oneOf(['onTouchStart', 'onTouchEnd', false])
} : void 0;

if (process.env.NODE_ENV !== 'production') {
  // eslint-disable-next-line
  ClickAwayListener['propTypes' + ''] = exactProp(ClickAwayListener.propTypes);
}

export default ClickAwayListener;
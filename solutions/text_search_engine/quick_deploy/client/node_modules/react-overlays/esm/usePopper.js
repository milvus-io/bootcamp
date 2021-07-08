import _extends from "@babel/runtime/helpers/esm/extends";
import PopperJS from 'popper.js';
import { useCallback, useEffect, useRef, useState } from 'react';
var initialPopperStyles = {
  position: 'absolute',
  top: '0',
  left: '0',
  opacity: '0',
  pointerEvents: 'none'
};
var initialArrowStyles = {};
/**
 * Position an element relative some reference element using Popper.js
 *
 * @param {HTMLElement} referenceElement The element
 * @param {HTMLElement} popperElement
 * @param {Object}      options
 * @param {Object}      options.modifiers Popper.js modifiers
 * @param {Boolean}     options.enabled toggle the popper functionality on/off
 * @param {String}      options.placement The popper element placement relative to the reference element
 * @param {Boolean}     options.positionFixed use fixed positioning
 * @param {Boolean}     options.eventsEnabled have Popper listen on window resize events to reposition the element
 */

export default function usePopper(referenceElement, popperElement, _temp) {
  var _ref = _temp === void 0 ? {} : _temp,
      _ref$enabled = _ref.enabled,
      enabled = _ref$enabled === void 0 ? true : _ref$enabled,
      _ref$placement = _ref.placement,
      placement = _ref$placement === void 0 ? 'bottom' : _ref$placement,
      _ref$positionFixed = _ref.positionFixed,
      positionFixed = _ref$positionFixed === void 0 ? false : _ref$positionFixed,
      _ref$eventsEnabled = _ref.eventsEnabled,
      eventsEnabled = _ref$eventsEnabled === void 0 ? true : _ref$eventsEnabled,
      _ref$modifiers = _ref.modifiers,
      modifiers = _ref$modifiers === void 0 ? {} : _ref$modifiers;

  var popperInstanceRef = useRef();
  var hasArrow = !!(modifiers.arrow && modifiers.arrow.element);
  var scheduleUpdate = useCallback(function () {
    if (popperInstanceRef.current) {
      popperInstanceRef.current.scheduleUpdate();
    }
  }, []);

  var _useState = useState({
    placement: placement,
    scheduleUpdate: scheduleUpdate,
    outOfBoundaries: false,
    styles: initialPopperStyles,
    arrowStyles: initialArrowStyles
  }),
      state = _useState[0],
      setState = _useState[1]; // A placement difference in state means popper determined a new placement
  // apart from the props value. By the time the popper element is rendered with
  // the new position Popper has already measured it, if the place change triggers
  // a size change it will result in a misaligned popper. So we schedule an update to be sure.


  useEffect(function () {
    scheduleUpdate();
  }, [state.placement, scheduleUpdate]);
  /** Toggle Events */

  useEffect(function () {
    if (popperInstanceRef.current) {
      // eslint-disable-next-line no-unused-expressions
      eventsEnabled ? popperInstanceRef.current.enableEventListeners() : popperInstanceRef.current.disableEventListeners();
    }
  }, [eventsEnabled]);
  useEffect(function () {
    if (!enabled || referenceElement == null || popperElement == null) {
      return undefined;
    }

    var arrow = modifiers.arrow && _extends({}, modifiers.arrow, {
      element: modifiers.arrow.element
    });

    popperInstanceRef.current = new PopperJS(referenceElement, popperElement, {
      placement: placement,
      positionFixed: positionFixed,
      modifiers: _extends({}, modifiers, {
        arrow: arrow,
        applyStyle: {
          enabled: false
        },
        updateStateModifier: {
          enabled: true,
          order: 900,
          fn: function fn(data) {
            setState({
              scheduleUpdate: scheduleUpdate,
              styles: _extends({
                position: data.offsets.popper.position
              }, data.styles),
              arrowStyles: data.arrowStyles,
              outOfBoundaries: data.hide,
              placement: data.placement
            });
          }
        }
      })
    });
    return function () {
      if (popperInstanceRef.current !== null) {
        popperInstanceRef.current.destroy();
        popperInstanceRef.current = null;
      }
    }; // intentionally NOT re-running on new modifiers
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, placement, positionFixed, referenceElement, popperElement, hasArrow]);
  return state;
}
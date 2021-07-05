"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));

var _objectWithoutProperties2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutProperties"));

var _react = _interopRequireDefault(require("react"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _popper = _interopRequireDefault(require("popper.js"));

var _utils = require("@material-ui/utils");

var _styles = require("@material-ui/styles");

var _Portal = _interopRequireDefault(require("../Portal"));

var _createChainedFunction = _interopRequireDefault(require("../utils/createChainedFunction"));

var _setRef = _interopRequireDefault(require("../utils/setRef"));

var _useForkRef = _interopRequireDefault(require("../utils/useForkRef"));

var _ownerWindow = _interopRequireDefault(require("../utils/ownerWindow"));

function flipPlacement(placement, theme) {
  var direction = theme && theme.direction || 'ltr';

  if (direction === 'ltr') {
    return placement;
  }

  switch (placement) {
    case 'bottom-end':
      return 'bottom-start';

    case 'bottom-start':
      return 'bottom-end';

    case 'top-end':
      return 'top-start';

    case 'top-start':
      return 'top-end';

    default:
      return placement;
  }
}

function getAnchorEl(anchorEl) {
  return typeof anchorEl === 'function' ? anchorEl() : anchorEl;
}

var useEnhancedEffect = typeof window !== 'undefined' ? _react.default.useLayoutEffect : _react.default.useEffect;
var defaultPopperOptions = {};
/**
 * Poppers rely on the 3rd party library [Popper.js](https://github.com/FezVrasta/popper.js) for positioning.
 */

var Popper = _react.default.forwardRef(function Popper(props, ref) {
  var anchorEl = props.anchorEl,
      children = props.children,
      container = props.container,
      _props$disablePortal = props.disablePortal,
      disablePortal = _props$disablePortal === void 0 ? false : _props$disablePortal,
      _props$keepMounted = props.keepMounted,
      keepMounted = _props$keepMounted === void 0 ? false : _props$keepMounted,
      modifiers = props.modifiers,
      open = props.open,
      _props$placement = props.placement,
      initialPlacement = _props$placement === void 0 ? 'bottom' : _props$placement,
      _props$popperOptions = props.popperOptions,
      popperOptions = _props$popperOptions === void 0 ? defaultPopperOptions : _props$popperOptions,
      popperRefProp = props.popperRef,
      _props$transition = props.transition,
      transition = _props$transition === void 0 ? false : _props$transition,
      other = (0, _objectWithoutProperties2.default)(props, ["anchorEl", "children", "container", "disablePortal", "keepMounted", "modifiers", "open", "placement", "popperOptions", "popperRef", "transition"]);

  var tooltipRef = _react.default.useRef(null);

  var ownRef = (0, _useForkRef.default)(tooltipRef, ref);

  var popperRef = _react.default.useRef(null);

  var handlePopperRef = (0, _useForkRef.default)(popperRef, popperRefProp);

  var handlePopperRefRef = _react.default.useRef(handlePopperRef);

  useEnhancedEffect(function () {
    handlePopperRefRef.current = handlePopperRef;
  }, [handlePopperRef]);

  _react.default.useImperativeHandle(popperRefProp, function () {
    return popperRef.current;
  }, []);

  var _React$useState = _react.default.useState(true),
      exited = _React$useState[0],
      setExited = _React$useState[1];

  var theme = (0, _styles.useTheme)();
  var rtlPlacement = flipPlacement(initialPlacement, theme);
  /**
   * placement initialized from prop but can change during lifetime if modifiers.flip.
   * modifiers.flip is essentially a flip for controlled/uncontrolled behavior
   */

  var _React$useState2 = _react.default.useState(rtlPlacement),
      placement = _React$useState2[0],
      setPlacement = _React$useState2[1];

  _react.default.useEffect(function () {
    if (popperRef.current) {
      popperRef.current.update();
    }
  });

  var handleOpen = _react.default.useCallback(function () {
    if (!tooltipRef.current || !anchorEl || !open) {
      return;
    }

    if (popperRef.current) {
      popperRef.current.destroy();
      handlePopperRefRef.current(null);
    }

    var handlePopperUpdate = function handlePopperUpdate(data) {
      setPlacement(data.placement);
    };

    var resolvedAnchorEl = getAnchorEl(anchorEl);

    if (process.env.NODE_ENV !== 'production') {
      var containerWindow = (0, _ownerWindow.default)(resolvedAnchorEl);

      if (resolvedAnchorEl instanceof containerWindow.Element) {
        var box = resolvedAnchorEl.getBoundingClientRect();

        if (process.env.NODE_ENV !== 'test' && box.top === 0 && box.left === 0 && box.right === 0 && box.bottom === 0) {
          console.warn(['Material-UI: the `anchorEl` prop provided to the component is invalid.', 'The anchor element should be part of the document layout.', "Make sure the element is present in the document or that it's not display none."].join('\n'));
        }
      }
    }

    var popper = new _popper.default(getAnchorEl(anchorEl), tooltipRef.current, (0, _extends2.default)({
      placement: rtlPlacement
    }, popperOptions, {
      modifiers: (0, _extends2.default)({}, disablePortal ? {} : {
        // It's using scrollParent by default, we can use the viewport when using a portal.
        preventOverflow: {
          boundariesElement: 'window'
        }
      }, {}, modifiers, {}, popperOptions.modifiers),
      // We could have been using a custom modifier like react-popper is doing.
      // But it seems this is the best public API for this use case.
      onCreate: (0, _createChainedFunction.default)(handlePopperUpdate, popperOptions.onCreate),
      onUpdate: (0, _createChainedFunction.default)(handlePopperUpdate, popperOptions.onUpdate)
    }));
    handlePopperRefRef.current(popper);
  }, [anchorEl, disablePortal, modifiers, open, rtlPlacement, popperOptions]);

  var handleRef = _react.default.useCallback(function (node) {
    (0, _setRef.default)(ownRef, node);
    handleOpen();
  }, [ownRef, handleOpen]);

  var handleEnter = function handleEnter() {
    setExited(false);
  };

  var handleClose = function handleClose() {
    if (!popperRef.current) {
      return;
    }

    popperRef.current.destroy();
    handlePopperRefRef.current(null);
  };

  var handleExited = function handleExited() {
    setExited(true);
    handleClose();
  };

  _react.default.useEffect(function () {
    // Let's update the popper position.
    handleOpen();
  }, [handleOpen]);

  _react.default.useEffect(function () {
    return function () {
      handleClose();
    };
  }, []);

  _react.default.useEffect(function () {
    if (!open && !transition) {
      // Otherwise handleExited will call this.
      handleClose();
    }
  }, [open, transition]);

  if (!keepMounted && !open && (!transition || exited)) {
    return null;
  }

  var childProps = {
    placement: placement
  };

  if (transition) {
    childProps.TransitionProps = {
      in: open,
      onEnter: handleEnter,
      onExited: handleExited
    };
  }

  return _react.default.createElement(_Portal.default, {
    disablePortal: disablePortal,
    container: container
  }, _react.default.createElement("div", (0, _extends2.default)({
    ref: handleRef,
    role: "tooltip"
  }, other, {
    style: (0, _extends2.default)({
      // Prevents scroll issue, waiting for Popper.js to add this style once initiated.
      position: 'fixed',
      // Fix Popper.js display issue
      top: 0,
      left: 0
    }, other.style)
  }), typeof children === 'function' ? children(childProps) : children));
});

process.env.NODE_ENV !== "production" ? Popper.propTypes = {
  /**
   * This is the reference element, or a function that returns the reference element,
   * that may be used to set the position of the popover.
   * The return value will passed as the reference object of the Popper
   * instance.
   *
   * The reference element should be an HTML Element instance or a referenceObject:
   * https://popper.js.org/popper-documentation.html#referenceObject.
   */
  anchorEl: (0, _utils.chainPropTypes)(_propTypes.default.oneOfType([_propTypes.default.object, _propTypes.default.func]), function (props) {
    if (props.open) {
      var resolvedAnchorEl = getAnchorEl(props.anchorEl);
      var containerWindow = (0, _ownerWindow.default)(resolvedAnchorEl);

      if (resolvedAnchorEl instanceof containerWindow.Element) {
        var box = resolvedAnchorEl.getBoundingClientRect();

        if (process.env.NODE_ENV !== 'test' && box.top === 0 && box.left === 0 && box.right === 0 && box.bottom === 0) {
          return new Error(['Material-UI: the `anchorEl` prop provided to the component is invalid.', 'The anchor element should be part of the document layout.', "Make sure the element is present in the document or that it's not display none."].join('\n'));
        }
      } else if (!resolvedAnchorEl || typeof resolvedAnchorEl.clientWidth !== 'number' || typeof resolvedAnchorEl.clientHeight !== 'number' || typeof resolvedAnchorEl.getBoundingClientRect !== 'function') {
        return new Error(['Material-UI: the `anchorEl` prop provided to the component is invalid.', 'It should be an HTML Element instance or a referenceObject:', 'https://popper.js.org/popper-documentation.html#referenceObject.'].join('\n'));
      }
    }

    return null;
  }),

  /**
   * Popper render function or node.
   */
  children: _propTypes.default.oneOfType([_propTypes.default.node, _propTypes.default.func]).isRequired,

  /**
   * A node, component instance, or function that returns either.
   * The `container` will passed to the Modal component.
   * By default, it uses the body of the anchorEl's top-level document object,
   * so it's simply `document.body` most of the time.
   */
  container: _propTypes.default.oneOfType([_propTypes.default.object, _propTypes.default.func]),

  /**
   * Disable the portal behavior.
   * The children stay within it's parent DOM hierarchy.
   */
  disablePortal: _propTypes.default.bool,

  /**
   * Always keep the children in the DOM.
   * This prop can be useful in SEO situation or
   * when you want to maximize the responsiveness of the Popper.
   */
  keepMounted: _propTypes.default.bool,

  /**
   * Popper.js is based on a "plugin-like" architecture,
   * most of its features are fully encapsulated "modifiers".
   *
   * A modifier is a function that is called each time Popper.js needs to
   * compute the position of the popper.
   * For this reason, modifiers should be very performant to avoid bottlenecks.
   * To learn how to create a modifier, [read the modifiers documentation](https://github.com/FezVrasta/popper.js/blob/master/docs/_includes/popper-documentation.md#modifiers--object).
   */
  modifiers: _propTypes.default.object,

  /**
   * If `true`, the popper is visible.
   */
  open: _propTypes.default.bool.isRequired,

  /**
   * Popper placement.
   */
  placement: _propTypes.default.oneOf(['bottom-end', 'bottom-start', 'bottom', 'left-end', 'left-start', 'left', 'right-end', 'right-start', 'right', 'top-end', 'top-start', 'top']),

  /**
   * Options provided to the [`popper.js`](https://github.com/FezVrasta/popper.js) instance.
   */
  popperOptions: _propTypes.default.object,

  /**
   * A ref that points to the used popper instance.
   */
  popperRef: _utils.refType,

  /**
   * Help supporting a react-transition-group/Transition component.
   */
  transition: _propTypes.default.bool
} : void 0;
var _default = Popper;
exports.default = _default;
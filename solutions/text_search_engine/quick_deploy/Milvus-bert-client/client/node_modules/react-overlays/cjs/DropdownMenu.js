"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

exports.__esModule = true;
exports.useDropdownMenu = useDropdownMenu;
exports["default"] = void 0;

var _objectWithoutPropertiesLoose2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutPropertiesLoose"));

var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));

var _propTypes = _interopRequireDefault(require("prop-types"));

var _react = require("react");

var _useCallbackRef2 = _interopRequireDefault(require("@restart/hooks/useCallbackRef"));

var _DropdownContext = _interopRequireDefault(require("./DropdownContext"));

var _usePopper = _interopRequireDefault(require("./usePopper"));

var _useRootClose = _interopRequireDefault(require("./useRootClose"));

function useDropdownMenu(options) {
  if (options === void 0) {
    options = {};
  }

  var context = (0, _react.useContext)(_DropdownContext["default"]);

  var _useCallbackRef = (0, _useCallbackRef2["default"])(),
      arrowElement = _useCallbackRef[0],
      attachArrowRef = _useCallbackRef[1];

  var hasShownRef = (0, _react.useRef)(false);
  var _options = options,
      flip = _options.flip,
      rootCloseEvent = _options.rootCloseEvent,
      _options$popperConfig = _options.popperConfig,
      popperConfig = _options$popperConfig === void 0 ? {} : _options$popperConfig,
      _options$usePopper = _options.usePopper,
      shouldUsePopper = _options$usePopper === void 0 ? true : _options$usePopper;
  var show = context.show == null ? options.show : context.show;
  var alignEnd = context.alignEnd == null ? options.alignEnd : context.alignEnd;

  if (show && !hasShownRef.current) {
    hasShownRef.current = true;
  }

  var handleClose = function handleClose(e) {
    if (!context.toggle) return;
    context.toggle(false, e);
  };

  var drop = context.drop,
      setMenu = context.setMenu,
      menuElement = context.menuElement,
      toggleElement = context.toggleElement;
  var placement = alignEnd ? 'bottom-end' : 'bottom-start';
  if (drop === 'up') placement = alignEnd ? 'top-end' : 'top-start';else if (drop === 'right') placement = alignEnd ? 'right-end' : 'right-start';else if (drop === 'left') placement = alignEnd ? 'left-end' : 'left-start';
  var popper = (0, _usePopper["default"])(toggleElement, menuElement, {
    placement: placement,
    enabled: !!(shouldUsePopper && show),
    eventsEnabled: !!show,
    modifiers: (0, _extends2["default"])({
      flip: {
        enabled: !!flip
      },
      arrow: (0, _extends2["default"])({}, popperConfig.modifiers && popperConfig.modifiers.arrow, {
        enabled: !!arrowElement,
        element: arrowElement
      })
    }, popperConfig.modifiers)
  });
  var menu = null;
  var menuProps = {
    ref: setMenu,
    'aria-labelledby': toggleElement && toggleElement.id
  };
  var childArgs = {
    show: show,
    alignEnd: alignEnd,
    hasShown: hasShownRef.current,
    close: handleClose
  };

  if (!shouldUsePopper) {
    menu = (0, _extends2["default"])({}, childArgs, {
      props: menuProps
    });
  } else {
    menu = (0, _extends2["default"])({}, popper, {}, childArgs, {
      props: (0, _extends2["default"])({}, menuProps, {
        style: popper.styles
      }),
      arrowProps: {
        ref: attachArrowRef,
        style: popper.arrowStyles
      }
    });
  }

  (0, _useRootClose["default"])(menuElement, handleClose, {
    clickTrigger: rootCloseEvent,
    disabled: !(menu && show)
  });
  return menu;
}

var propTypes = {
  /**
   * A render prop that returns a Menu element. The `props`
   * argument should spread through to **a component that can accept a ref**.
   *
   * @type {Function ({
   *   show: boolean,
   *   alignEnd: boolean,
   *   close: (?SyntheticEvent) => void,
   *   placement: Placement,
   *   outOfBoundaries: ?boolean,
   *   scheduleUpdate: () => void,
   *   props: {
   *     ref: (?HTMLElement) => void,
   *     style: { [string]: string | number },
   *     aria-labelledby: ?string
   *   },
   *   arrowProps: {
   *     ref: (?HTMLElement) => void,
   *     style: { [string]: string | number },
   *   },
   * }) => React.Element}
   */
  children: _propTypes["default"].func.isRequired,

  /**
   * Controls the visible state of the menu, generally this is
   * provided by the parent `Dropdown` component,
   * but may also be specified as a prop directly.
   */
  show: _propTypes["default"].bool,

  /**
   * Aligns the dropdown menu to the 'end' of it's placement position.
   * Generally this is provided by the parent `Dropdown` component,
   * but may also be specified as a prop directly.
   */
  alignEnd: _propTypes["default"].bool,

  /**
   * Enables the Popper.js `flip` modifier, allowing the Dropdown to
   * automatically adjust it's placement in case of overlap with the viewport or toggle.
   * Refer to the [flip docs](https://popper.js.org/popper-documentation.html#modifiers..flip.enabled) for more info
   */
  flip: _propTypes["default"].bool,
  usePopper: _propTypes["default"].oneOf([true, false]),

  /**
   * A set of popper options and props passed directly to react-popper's Popper component.
   */
  popperConfig: _propTypes["default"].object,

  /**
   * Override the default event used by RootCloseWrapper.
   */
  rootCloseEvent: _propTypes["default"].string
};
var defaultProps = {
  usePopper: true
};

function DropdownMenu(_ref) {
  var children = _ref.children,
      options = (0, _objectWithoutPropertiesLoose2["default"])(_ref, ["children"]);
  var args = useDropdownMenu(options);
  return args.hasShown ? children(args) : null;
}

DropdownMenu.displayName = 'ReactOverlaysDropdownMenu';
DropdownMenu.propTypes = propTypes;
DropdownMenu.defaultProps = defaultProps;
/** @component */

var _default = DropdownMenu;
exports["default"] = _default;
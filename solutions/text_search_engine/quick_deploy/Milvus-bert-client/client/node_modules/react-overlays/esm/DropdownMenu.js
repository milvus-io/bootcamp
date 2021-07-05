import _objectWithoutPropertiesLoose from "@babel/runtime/helpers/esm/objectWithoutPropertiesLoose";
import _extends from "@babel/runtime/helpers/esm/extends";
import PropTypes from 'prop-types';
import { useContext, useRef } from 'react';
import useCallbackRef from '@restart/hooks/useCallbackRef';
import DropdownContext from './DropdownContext';
import usePopper from './usePopper';
import useRootClose from './useRootClose';
export function useDropdownMenu(options) {
  if (options === void 0) {
    options = {};
  }

  var context = useContext(DropdownContext);

  var _useCallbackRef = useCallbackRef(),
      arrowElement = _useCallbackRef[0],
      attachArrowRef = _useCallbackRef[1];

  var hasShownRef = useRef(false);
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
  var popper = usePopper(toggleElement, menuElement, {
    placement: placement,
    enabled: !!(shouldUsePopper && show),
    eventsEnabled: !!show,
    modifiers: _extends({
      flip: {
        enabled: !!flip
      },
      arrow: _extends({}, popperConfig.modifiers && popperConfig.modifiers.arrow, {
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
    menu = _extends({}, childArgs, {
      props: menuProps
    });
  } else {
    menu = _extends({}, popper, {}, childArgs, {
      props: _extends({}, menuProps, {
        style: popper.styles
      }),
      arrowProps: {
        ref: attachArrowRef,
        style: popper.arrowStyles
      }
    });
  }

  useRootClose(menuElement, handleClose, {
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
  children: PropTypes.func.isRequired,

  /**
   * Controls the visible state of the menu, generally this is
   * provided by the parent `Dropdown` component,
   * but may also be specified as a prop directly.
   */
  show: PropTypes.bool,

  /**
   * Aligns the dropdown menu to the 'end' of it's placement position.
   * Generally this is provided by the parent `Dropdown` component,
   * but may also be specified as a prop directly.
   */
  alignEnd: PropTypes.bool,

  /**
   * Enables the Popper.js `flip` modifier, allowing the Dropdown to
   * automatically adjust it's placement in case of overlap with the viewport or toggle.
   * Refer to the [flip docs](https://popper.js.org/popper-documentation.html#modifiers..flip.enabled) for more info
   */
  flip: PropTypes.bool,
  usePopper: PropTypes.oneOf([true, false]),

  /**
   * A set of popper options and props passed directly to react-popper's Popper component.
   */
  popperConfig: PropTypes.object,

  /**
   * Override the default event used by RootCloseWrapper.
   */
  rootCloseEvent: PropTypes.string
};
var defaultProps = {
  usePopper: true
};

function DropdownMenu(_ref) {
  var children = _ref.children,
      options = _objectWithoutPropertiesLoose(_ref, ["children"]);

  var args = useDropdownMenu(options);
  return args.hasShown ? children(args) : null;
}

DropdownMenu.displayName = 'ReactOverlaysDropdownMenu';
DropdownMenu.propTypes = propTypes;
DropdownMenu.defaultProps = defaultProps;
/** @component */

export default DropdownMenu;
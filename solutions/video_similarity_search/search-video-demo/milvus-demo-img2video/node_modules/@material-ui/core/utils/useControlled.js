"use strict";

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = useControlled;

var _react = _interopRequireDefault(require("react"));

/* eslint-disable react-hooks/rules-of-hooks, react-hooks/exhaustive-deps */
function useControlled(_ref) {
  var controlled = _ref.controlled,
      defaultProp = _ref.default,
      name = _ref.name;

  var _React$useRef = _react.default.useRef(controlled !== undefined),
      isControlled = _React$useRef.current;

  var _React$useState = _react.default.useState(defaultProp),
      valueState = _React$useState[0],
      setValue = _React$useState[1];

  var value = isControlled ? controlled : valueState;

  if (process.env.NODE_ENV !== 'production') {
    _react.default.useEffect(function () {
      if (isControlled !== (controlled !== undefined)) {
        console.error(["Material-UI: A component is changing ".concat(isControlled ? 'a ' : 'an un', "controlled ").concat(name, " to be ").concat(isControlled ? 'un' : '', "controlled."), 'Elements should not switch from uncontrolled to controlled (or vice versa).', "Decide between using a controlled or uncontrolled ".concat(name, " ") + 'element for the lifetime of the component.', 'More info: https://fb.me/react-controlled-components'].join('\n'));
      }
    }, [controlled]);

    var _React$useRef2 = _react.default.useRef(defaultProp),
        defaultValue = _React$useRef2.current;

    _react.default.useEffect(function () {
      if (defaultValue !== defaultProp) {
        console.error(["Material-UI: A component is changing the default value of an uncontrolled ".concat(name, " after being initialized. ") + "To suppress this warning opt to use a controlled ".concat(name, ".")].join('\n'));
      }
    }, [JSON.stringify(defaultProp)]);
  }

  var setValueIfUncontrolled = _react.default.useCallback(function (newValue) {
    if (!isControlled) {
      setValue(newValue);
    }
  }, []);

  return [value, setValueIfUncontrolled];
}
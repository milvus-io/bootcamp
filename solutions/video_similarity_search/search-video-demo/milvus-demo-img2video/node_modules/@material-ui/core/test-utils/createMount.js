"use strict";

var _interopRequireWildcard = require("@babel/runtime/helpers/interopRequireWildcard");

var _interopRequireDefault = require("@babel/runtime/helpers/interopRequireDefault");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = createMount;

var _extends2 = _interopRequireDefault(require("@babel/runtime/helpers/extends"));

var _objectWithoutProperties2 = _interopRequireDefault(require("@babel/runtime/helpers/objectWithoutProperties"));

var _classCallCheck2 = _interopRequireDefault(require("@babel/runtime/helpers/classCallCheck"));

var _createClass2 = _interopRequireDefault(require("@babel/runtime/helpers/createClass"));

var _possibleConstructorReturn2 = _interopRequireDefault(require("@babel/runtime/helpers/possibleConstructorReturn"));

var _getPrototypeOf2 = _interopRequireDefault(require("@babel/runtime/helpers/getPrototypeOf"));

var _inherits2 = _interopRequireDefault(require("@babel/runtime/helpers/inherits"));

var _react = _interopRequireDefault(require("react"));

var _reactDom = _interopRequireDefault(require("react-dom"));

var PropTypes = _interopRequireWildcard(require("prop-types"));

var _enzyme = require("enzyme");

/**
 * Can't just mount <React.Fragment>{node}</React.Fragment>
 * because that swallows wrapper.setProps
 *
 * why class component:
 * https://github.com/airbnb/enzyme/issues/2043
 */
// eslint-disable-next-line react/prefer-stateless-function
var Mode =
/*#__PURE__*/
function (_React$Component) {
  (0, _inherits2.default)(Mode, _React$Component);

  function Mode() {
    (0, _classCallCheck2.default)(this, Mode);
    return (0, _possibleConstructorReturn2.default)(this, (0, _getPrototypeOf2.default)(Mode).apply(this, arguments));
  }

  (0, _createClass2.default)(Mode, [{
    key: "render",
    value: function render() {
      // Excess props will come from e.g. enzyme setProps
      var _this$props = this.props,
          __element = _this$props.__element,
          __strict = _this$props.__strict,
          other = (0, _objectWithoutProperties2.default)(_this$props, ["__element", "__strict"]);
      var Component = __strict ? _react.default.StrictMode : _react.default.Fragment;
      return _react.default.createElement(Component, null, _react.default.cloneElement(__element, other));
    }
  }]);
  return Mode;
}(_react.default.Component); // Generate an enhanced mount function.


process.env.NODE_ENV !== "production" ? Mode.propTypes = {
  /**
   * this is essentially children. However we can't use children because then
   * using `wrapper.setProps({ children })` would work differently if this component
   * would be the root.
   */
  __element: PropTypes.element.isRequired,
  __strict: PropTypes.bool.isRequired
} : void 0;

function createMount() {
  var options = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
  var _options$mount = options.mount,
      mount = _options$mount === void 0 ? _enzyme.mount : _options$mount,
      globalStrict = options.strict,
      globalEnzymeOptions = (0, _objectWithoutProperties2.default)(options, ["mount", "strict"]);
  var attachTo = document.createElement('div');
  attachTo.className = 'app';
  attachTo.setAttribute('id', 'app');
  document.body.insertBefore(attachTo, document.body.firstChild);

  var mountWithContext = function mountWithContext(node) {
    var localOptions = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
    var _localOptions$disable = localOptions.disableUnnmount,
        disableUnnmount = _localOptions$disable === void 0 ? false : _localOptions$disable,
        _localOptions$strict = localOptions.strict,
        strict = _localOptions$strict === void 0 ? globalStrict : _localOptions$strict,
        localEnzymeOptions = (0, _objectWithoutProperties2.default)(localOptions, ["disableUnnmount", "strict"]);

    if (!disableUnnmount) {
      _reactDom.default.unmountComponentAtNode(attachTo);
    } // some tests require that no other components are in the tree
    // e.g. when doing .instance(), .state() etc.


    return mount(strict == null ? node : _react.default.createElement(Mode, {
      __element: node,
      __strict: Boolean(strict)
    }), (0, _extends2.default)({
      attachTo: attachTo
    }, globalEnzymeOptions, {}, localEnzymeOptions));
  };

  mountWithContext.attachTo = attachTo;

  mountWithContext.cleanUp = function () {
    _reactDom.default.unmountComponentAtNode(attachTo);

    attachTo.parentElement.removeChild(attachTo);
  };

  return mountWithContext;
}
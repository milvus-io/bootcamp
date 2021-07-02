"use strict";

exports.__esModule = true;
exports.default = exports.ObservableMap = void 0;

var _useForceUpdate = _interopRequireDefault(require("./useForceUpdate"));

var _useStableMemo = _interopRequireDefault(require("./useStableMemo"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _inheritsLoose(subClass, superClass) { subClass.prototype = Object.create(superClass.prototype); subClass.prototype.constructor = subClass; subClass.__proto__ = superClass; }

function _wrapNativeSuper(Class) { var _cache = typeof Map === "function" ? new Map() : undefined; _wrapNativeSuper = function _wrapNativeSuper(Class) { if (Class === null || !_isNativeFunction(Class)) return Class; if (typeof Class !== "function") { throw new TypeError("Super expression must either be null or a function"); } if (typeof _cache !== "undefined") { if (_cache.has(Class)) return _cache.get(Class); _cache.set(Class, Wrapper); } function Wrapper() { return _construct(Class, arguments, _getPrototypeOf(this).constructor); } Wrapper.prototype = Object.create(Class.prototype, { constructor: { value: Wrapper, enumerable: false, writable: true, configurable: true } }); return _setPrototypeOf(Wrapper, Class); }; return _wrapNativeSuper(Class); }

function isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Date.prototype.toString.call(Reflect.construct(Date, [], function () {})); return true; } catch (e) { return false; } }

function _construct(Parent, args, Class) { if (isNativeReflectConstruct()) { _construct = Reflect.construct; } else { _construct = function _construct(Parent, args, Class) { var a = [null]; a.push.apply(a, args); var Constructor = Function.bind.apply(Parent, a); var instance = new Constructor(); if (Class) _setPrototypeOf(instance, Class.prototype); return instance; }; } return _construct.apply(null, arguments); }

function _isNativeFunction(fn) { return Function.toString.call(fn).indexOf("[native code]") !== -1; }

function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }

function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }

var ObservableMap =
/*#__PURE__*/
function (_Map) {
  _inheritsLoose(ObservableMap, _Map);

  function ObservableMap(listener, init) {
    var _this;

    _this = _Map.call(this, init) || this;
    _this.listener = listener;
    return _this;
  }

  var _proto = ObservableMap.prototype;

  _proto.set = function set(key, value) {
    _Map.prototype.set.call(this, key, value); // When initializing the Map, the base Map calls this.set() before the
    // listener is assigned so it will be undefined


    if (this.listener) this.listener(this);
    return this;
  };

  _proto.delete = function _delete(key) {
    var result = _Map.prototype.delete.call(this, key);

    this.listener(this);
    return result;
  };

  _proto.clear = function clear() {
    _Map.prototype.clear.call(this);

    this.listener(this);
  };

  return ObservableMap;
}(
/*#__PURE__*/
_wrapNativeSuper(Map));
/**
 * Create and return a [Map](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map) that triggers rerenders when it's updated.
 *
 * ```tsx
 * const customerAges = useMap<number>([
 *  ['john', 24],
 *  ['betsy', 25]
 * ]);
 *
 * return (
 *  <>
 *    {Array.from(ids, ([name, age]) => (
 *      <div>
 *        {name}: {age}. <button onClick={() => ids.delete(name)}>X</button>
 *      </div>
 *    )}
 *  </>
 * )
 * ```
 *
 * @param init initial Map entries
 */


exports.ObservableMap = ObservableMap;

function useMap(init) {
  var forceUpdate = (0, _useForceUpdate.default)();
  return (0, _useStableMemo.default)(function () {
    return new ObservableMap(forceUpdate, init);
  }, []);
}

var _default = useMap;
exports.default = _default;
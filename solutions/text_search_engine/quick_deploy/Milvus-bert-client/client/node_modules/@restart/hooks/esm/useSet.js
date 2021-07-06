function _inheritsLoose(subClass, superClass) { subClass.prototype = Object.create(superClass.prototype); subClass.prototype.constructor = subClass; subClass.__proto__ = superClass; }

function _wrapNativeSuper(Class) { var _cache = typeof Map === "function" ? new Map() : undefined; _wrapNativeSuper = function _wrapNativeSuper(Class) { if (Class === null || !_isNativeFunction(Class)) return Class; if (typeof Class !== "function") { throw new TypeError("Super expression must either be null or a function"); } if (typeof _cache !== "undefined") { if (_cache.has(Class)) return _cache.get(Class); _cache.set(Class, Wrapper); } function Wrapper() { return _construct(Class, arguments, _getPrototypeOf(this).constructor); } Wrapper.prototype = Object.create(Class.prototype, { constructor: { value: Wrapper, enumerable: false, writable: true, configurable: true } }); return _setPrototypeOf(Wrapper, Class); }; return _wrapNativeSuper(Class); }

function isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Date.prototype.toString.call(Reflect.construct(Date, [], function () {})); return true; } catch (e) { return false; } }

function _construct(Parent, args, Class) { if (isNativeReflectConstruct()) { _construct = Reflect.construct; } else { _construct = function _construct(Parent, args, Class) { var a = [null]; a.push.apply(a, args); var Constructor = Function.bind.apply(Parent, a); var instance = new Constructor(); if (Class) _setPrototypeOf(instance, Class.prototype); return instance; }; } return _construct.apply(null, arguments); }

function _isNativeFunction(fn) { return Function.toString.call(fn).indexOf("[native code]") !== -1; }

function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf || function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }

function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }

import useForceUpdate from './useForceUpdate';
import useStableMemo from './useStableMemo';
export var ObservableSet =
/*#__PURE__*/
function (_Set) {
  _inheritsLoose(ObservableSet, _Set);

  function ObservableSet(listener, init) {
    var _this;

    _this = _Set.call(this, init) || this;
    _this.listener = listener;
    return _this;
  }

  var _proto = ObservableSet.prototype;

  _proto.add = function add(value) {
    _Set.prototype.add.call(this, value); // When initializing the Set, the base Set calls this.add() before the
    // listener is assigned so it will be undefined


    if (this.listener) this.listener(this);
    return this;
  };

  _proto.delete = function _delete(value) {
    var result = _Set.prototype.delete.call(this, value);

    this.listener(this);
    return result;
  };

  _proto.clear = function clear() {
    _Set.prototype.clear.call(this);

    this.listener(this);
  };

  return ObservableSet;
}(
/*#__PURE__*/
_wrapNativeSuper(Set));
/**
 * Create and return a [Set](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set) that triggers rerenders when it's updated.
 *
 * ```ts
 * const ids = useSet<number>([1,2,3,4]);
 *
 * return (
 *  <>
 *    {Array.from(ids, id => (
 *      <div>
 *        id: {id}. <button onClick={() => ids.delete(id)}>X</button>
 *      </div>
 *    )}
 *  </>
 * )
 * ```
 *
 * @param init initial Set values
 */

function useSet(init) {
  var forceUpdate = useForceUpdate();
  return useStableMemo(function () {
    return new ObservableSet(forceUpdate, init);
  }, []);
}

export default useSet;
function unstable_getIn(object, path, defaultValue) {
  if (typeof path === "string") {
    return object[path] == null ? defaultValue : object[path];
  }

  var result = object;

  for (var _iterator = path, _isArray = Array.isArray(_iterator), _i = 0, _iterator = _isArray ? _iterator : _iterator[Symbol.iterator]();;) {
    var _ref;

    if (_isArray) {
      if (_i >= _iterator.length) break;
      _ref = _iterator[_i++];
    } else {
      _i = _iterator.next();
      if (_i.done) break;
      _ref = _i.value;
    }

    var key = _ref;
    if (!(key in result)) return defaultValue;
    result = result[key];
  }

  if (result == null) {
    return defaultValue;
  }

  return result;
}

export { unstable_getIn };

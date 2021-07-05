'use strict';

var path       = require('path'),
    fs         = require('fs'),
    objectPath = require('object-path');

var getContextDirectory = require('./get-context-directory');

/**
 * Infer the compilation output directory from options.
 * Relative paths are resolved against the compilation context (or process.cwd() where not specified).
 * @this {{options: object}} A loader or compilation
 * @returns {undefined|string} The output path string, where defined
 */
function getOutputDirectory() {
  /* jshint validthis:true */
  var base    = objectPath.get(this, 'options.output.directory'),
      absBase = !!base && path.resolve(getContextDirectory.call(this), base),
      isValid = !!absBase && fs.existsSync(absBase) && fs.statSync(absBase).isDirectory();
  return isValid ? absBase : undefined;
}

module.exports = getOutputDirectory;
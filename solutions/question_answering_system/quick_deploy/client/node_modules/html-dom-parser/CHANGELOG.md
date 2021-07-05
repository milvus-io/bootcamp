# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

### [0.2.3](https://github.com/remarkablemark/html-dom-parser/compare/v0.2.2...v0.2.3) (2019-11-04)


### Bug Fixes

* **lib:** improve head and body regex in `domparser.js` ([457bb58](https://github.com/remarkablemark/html-dom-parser/commit/457bb58)), closes [#18](https://github.com/remarkablemark/html-dom-parser/issues/18)


### Build System

* **package:** save commitlint, husky, and lint-staged to devDeps ([3b0ce91](https://github.com/remarkablemark/html-dom-parser/commit/3b0ce91))
* **package:** update `eslint` and install `prettier` and plugin ([b7a6b81](https://github.com/remarkablemark/html-dom-parser/commit/b7a6b81))
* **package:** update `webpack` and save `webpack-cli` ([908e56d](https://github.com/remarkablemark/html-dom-parser/commit/908e56d))
* **package:** update dependencies and devDependencies ([a9016be](https://github.com/remarkablemark/html-dom-parser/commit/a9016be))


### Tests

* **server:** remove skipped test ([a4c1057](https://github.com/remarkablemark/html-dom-parser/commit/a4c1057))
* refactor tests to ES6 ([d5255a5](https://github.com/remarkablemark/html-dom-parser/commit/d5255a5))
* **cases:** add empty string test case to `html.js` ([25d7e8a](https://github.com/remarkablemark/html-dom-parser/commit/25d7e8a))
* **cases:** add more special test cases to `html.js` ([6fdf2ea](https://github.com/remarkablemark/html-dom-parser/commit/6fdf2ea))
* **cases:** refactor test cases and move html data to its own file ([e4fcb09](https://github.com/remarkablemark/html-dom-parser/commit/e4fcb09))
* **cases:** remove unnecessary try/catch wrapper to fix lint error ([ca8175e](https://github.com/remarkablemark/html-dom-parser/commit/ca8175e))
* **cases:** skip html test cases that PhantomJS does not support ([d095d29](https://github.com/remarkablemark/html-dom-parser/commit/d095d29))
* **cases:** update `complex.html` ([1418775](https://github.com/remarkablemark/html-dom-parser/commit/1418775))
* **client:** add tests for client parser that will be run by karma ([a0c58aa](https://github.com/remarkablemark/html-dom-parser/commit/a0c58aa))
* **helpers:** create `index.js` which exports helpers ([a9255d5](https://github.com/remarkablemark/html-dom-parser/commit/a9255d5))
* **helpers:** move helper that tests for errors to separate file ([f2e6312](https://github.com/remarkablemark/html-dom-parser/commit/f2e6312))
* **helpers:** refactor and move `runTests` to its own file ([8e30784](https://github.com/remarkablemark/html-dom-parser/commit/8e30784))
* **server:** add tests that spy and mock htmlparser2 and domhandler ([61075a1](https://github.com/remarkablemark/html-dom-parser/commit/61075a1))
* **server:** move `html-to-dom-server.js` to `server` directory ([3684dac](https://github.com/remarkablemark/html-dom-parser/commit/3684dac))



## [0.2.2](https://github.com/remarkablemark/html-dom-parser/compare/v0.2.1...v0.2.2) (2019-06-07)


### Bug Fixes

* **utilities:** do not lowercase case-sensitive SVG tags ([4083004](https://github.com/remarkablemark/html-dom-parser/commit/4083004))


### Performance Improvements

* **utilities:** optimize case-sensitive tag replace with hash map ([6aa06ee](https://github.com/remarkablemark/html-dom-parser/commit/6aa06ee))



## [0.2.1](https://github.com/remarkablemark/html-dom-parser/compare/v0.2.0...v0.2.1) (2019-04-03)



# [0.2.0](https://github.com/remarkablemark/html-dom-parser/compare/v0.1.3...v0.2.0) (2019-04-01)


### Features

* **types:** add TypeScript decelerations ([b52d52f](https://github.com/remarkablemark/html-dom-parser/commit/b52d52f))



## [0.1.3](https://github.com/remarkablemark/html-dom-parser/compare/v0.1.2...v0.1.3) - 2018-02-20
### Fixed
- Fix regular expression vulnerability (#8)
  - Regex has potential for catastrophic backtracking
  - Credit goes to @davisjam for discovering it

### Changed
- Refactored and updated tests (#8)

## [0.1.2](https://github.com/remarkablemark/html-dom-parser/compare/v0.1.1...v0.1.2) - 2017-09-30
### Added
- Create helper `isIE()` in utilities (#7)

### Fixed
- Fix client parser in IE/IE9 (#6, #7)

### Changed
- Upgrade `mocha@3.4.2` and `webpack@2.6.1` (#5)
- npm script `build` runs both `build:min` and `build:unmin` (#5)

## [0.1.1](https://github.com/remarkablemark/html-dom-parser/compare/v0.1.0...v0.1.1) - 2017-06-26
### Added
- CHANGELOG with previous releases backfilled

### Fixed
- Fix client parser on IE by specifying required parameter for `createHTMLDocument` (#4)

## [0.1.0](https://github.com/remarkablemark/html-dom-parser/compare/v0.0.2...v0.1.0) - 2017-06-17
### Changed
- Improve, refactor, and optimize client parser
  - Use `template`, `DOMImplementation`, and/or `DOMParser`

## [0.0.2](https://github.com/remarkablemark/html-dom-parser/compare/v0.0.1...v0.0.2) - 2016-10-10
### Added
- Create npm scripts for prepublish

### Changed
- Change webpack to build to UMD target
- Update README installation and usage instructions

## [0.0.1](https://github.com/remarkablemark/html-dom-parser/tree/v0.0.1) - 2016-10-10
### Added
- Server parser
  - Wrapper for `htmlparser2.parseDOM`
- Client parser
  - Uses DOM API to mimic server parser output
  - Build client library with webpack
- Add README, tests, and other necessary files

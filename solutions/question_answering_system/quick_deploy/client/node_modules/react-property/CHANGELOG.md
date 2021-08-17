# Change Log

All notable changes to this project will be documented in this file.
See [Conventional Commits](https://conventionalcommits.org) for commit guidelines.

## [1.0.1](https://github.com/remarkablemark/react-dom-core/compare/react-property@1.0.0...react-property@1.0.1) (2019-07-09)

**Note:** Version bump only for package react-property





# [1.0.0](https://github.com/remarkablemark/react-dom-core/compare/react-property@0.1.0...react-property@1.0.0) (2019-07-09)


### Features

* **react-property:** add script to build injection json ([2de3da9](https://github.com/remarkablemark/react-dom-core/commit/2de3da9))
* **react-property:** consolidate injection logic to `index.js` ([167887f](https://github.com/remarkablemark/react-dom-core/commit/167887f))
* **react-property:** export html and svg property configs ([8f8b921](https://github.com/remarkablemark/react-dom-core/commit/8f8b921))
* **react-property:** rewrite build html script ([a44567a](https://github.com/remarkablemark/react-dom-core/commit/a44567a))
* **react-property:** rewrite build svg script ([eb7a59b](https://github.com/remarkablemark/react-dom-core/commit/eb7a59b))


### BREAKING CHANGES

* **react-property:** remove exports `HTMLDOMPropertyConfig` and
`SVGDOMPropertyConfig` and consolidate the properties in
`properties`.

As a result of this change, `src/` directory is removed since all
the injection logic is handled in `index.js` and the npm script
`copy` is removed as well.





# 0.1.0 (2019-07-05)


### Features

* **react-property:** add `index.js` that imports all files in lib ([08d4ba2](https://github.com/remarkablemark/react-dom-core/commit/08d4ba2))
* **react-property:** add build script for SVG DOM config ([182326a](https://github.com/remarkablemark/react-dom-core/commit/182326a))
* **react-property:** add module that exports `isCustomAttribute` ([4213cbd](https://github.com/remarkablemark/react-dom-core/commit/4213cbd))
* **react-property:** add module that exports HTML DOM attribute map ([925db59](https://github.com/remarkablemark/react-dom-core/commit/925db59))
* **react-property:** add module that exports SVG DOM attribute map ([b6501b3](https://github.com/remarkablemark/react-dom-core/commit/b6501b3))
* **react-property:** add script that builds HTML attributes (JSON) ([c5349b8](https://github.com/remarkablemark/react-dom-core/commit/c5349b8))
* **react-property:** build HTML overloaded boolean properties JSON ([ab7e2c2](https://github.com/remarkablemark/react-dom-core/commit/ab7e2c2))
* **react-property:** build HTML props with boolean values (JSON) ([329593c](https://github.com/remarkablemark/react-dom-core/commit/329593c))
* **react-property:** throw error if mkdir fails in build scripts ([1566b2f](https://github.com/remarkablemark/react-dom-core/commit/1566b2f))
* **react-property:** update script to build HTML attr to prop JSON ([f09374d](https://github.com/remarkablemark/react-dom-core/commit/f09374d))

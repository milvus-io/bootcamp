## 0.3.0

### Minor Changes

- 7f1ada0: Internal polish

## 0.2.0

### Minor Changes

- eb86842: Add option to mock window.getComputedStyle

  This option has two use cases in mind:

  1. fake the style and assume everything is visible.
     This increases performance (window.getComputedStyle) is expensive) by not distinguishing between various levels of visual impairments. If one can't see the name with a screen reader then neither will a sighted user
  2. Wrap a cache provider around `window.getComputedStyle`. We don't implement any because the returned `CSSStyleDeclaration` is only live in a browser. `jsdom` does not implement live declarations.

### Bug Fixes

- Fix test name_heading-combobox ([#16](https://github.com/eps1lon/dom-accessibility-api/issues/16)) ([e969395](https://github.com/eps1lon/dom-accessibility-api/commit/e969395d8da637862993aeee0b86f379342d56f2))

### Features

- **name:** Consider prohibited naming ([#19](https://github.com/eps1lon/dom-accessibility-api/issues/19)) ([6692d6b](https://github.com/eps1lon/dom-accessibility-api/commit/6692d6bd86030da9b340b0895f623394b21e2656))
- Consider all cases of "name from content" ([#13](https://github.com/eps1lon/dom-accessibility-api/issues/13)) ([835cb76](https://github.com/eps1lon/dom-accessibility-api/commit/835cb76e7c1dd577af1fa891ad849385e58fcd56))
- Consider content from before and after pseudo elements ([#5](https://github.com/eps1lon/dom-accessibility-api/issues/5)) ([0987426](https://github.com/eps1lon/dom-accessibility-api/commit/0987426734cc7b980a8edf39435820a24ea2a162))
- Fork elementToRole from aria-query ([#7](https://github.com/eps1lon/dom-accessibility-api/issues/7)) ([fe4fab5](https://github.com/eps1lon/dom-accessibility-api/commit/fe4fab57786324705c4ac4434de8aabd3e7bbc09))

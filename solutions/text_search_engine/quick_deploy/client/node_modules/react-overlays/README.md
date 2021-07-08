# react-overlays

[![test status][test-badge]][actions]
[![deploy docs status][deploy-docs-badge]][actions]
[![codecov][codecov-badge]][codecov]
[![Netlify Status](https://api.netlify.com/api/v1/badges/e86fa356-4480-409e-9c24-52ea0660a923/deploy-status)](https://app.netlify.com/sites/react-overlays/deploys)

Utilities for creating robust overlay components

demos and docs at: https://react-bootstrap.github.io/react-overlays/modal/

## Install

```sh
npm install --save react-overlays
```

All of these utilities have been abstracted out of [react-bootstrap](https://github.com/react-bootstrap/react-bootstrap) in order to provide better access to the generic implementations of these commonly needed components. The included components are building blocks for creating more polished components. Everything is bring-your-own-styles, css or otherwise.

If you are looking for more complete overlays, modals, or tooltips--something you can use right out of the box--check out react-bootstrap, which is (or soon will be) built on using these components.

**note:** we are still in the process of abstracting out these components so the API's will probably change until we are sure that all of the bootstrap components can cleanly be implemented on top of them.

Pre `1.0.0` breaking changes happen on the `minor` bump while feature and patches accompany a `patch` bump.

[actions]: https://github.com/react-bootstrap/react-overlays/actions
[codecov]: https://codecov.io/gh/react-bootstrap/react-overlays
[codecov-badge]: https://codecov.io/gh/react-bootstrap/react-overlays/branch/master/graph/badge.svg
[test-badge]: https://github.com/react-bootstrap/react-overlays/workflows/Run%20Tests/badge.svg
[deploy-docs-badge]: https://github.com/react-bootstrap/react-overlays/workflows/Deploy%20Documentation/badge.svg

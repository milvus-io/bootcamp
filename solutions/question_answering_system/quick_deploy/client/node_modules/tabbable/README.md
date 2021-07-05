# tabbable

[![Build Status](https://travis-ci.org/davidtheclark/tabbable.svg?branch=master)](https://travis-ci.org/davidtheclark/tabbable)

Returns an array of all tabbable DOM nodes within a containing node, *in their actual tab order* (cf. [Sequential focus navigation and the tabindex attribute](http://www.w3.org/TR/html5/editing.html#sequential-focus-navigation-and-the-tabindex-attribute)).

This should include

- `<input>`s,
- `<select>`s,
- `<textarea>`s,
- `<button>`s,
- `<a>`s with `href` attributes *or* non-negative `tabindex`es,
- anything else with a non-negative `tabindex`

Any of the above will *not* be added to the array, though, if any of the following are also true about it:
- negative `tabindex`
- `disabled`
- either the node itself *or an ancestor of it* is hidden via `display: none` or `visibility: hidden`

**Note**: Though browsers allow tabbing into elements marked `contenteditable`, outstanding bugs in the `tabIndex` API prevents `tabbable` from registering them. If you have `contenteditable` elements that you need included in the array, you'll have to additionally specify `tabindex="0"`.

## Goals
- Accurate
- No dependencies
- Small
- Fast

## Browser Support

Basically IE9+.

Why? It uses [Element.querySelectorAll()](https://developer.mozilla.org/en-US/docs/Web/API/Element/querySelectorAll) and [Window.getComputedStyle()](https://developer.mozilla.org/en-US/docs/Web/API/Window/getComputedStyle).

## Installation

```
npm install tabbable
```

Dependencies: *none*.

You'll need to be compiling CommonJS (via browserify or webpack).

## API

```
tabbable(rootNode, [options])
```

Returns an array of ordered tabbable node within the `rootNode`.

Summary of ordering principles:
- First include any nodes with positive `tabindex` attributes (1 or higher), ordered by ascending `tabindex` and source order.
- Then include any nodes with a zero `tabindex` and any element that by default receives focus (listed above) and does not have a positive `tabindex` set, in source order.

### rootNode

Type: `Node`. **Required.**

### options

#### includeContainer

Type: `boolean`. Default: `false`.

If set to `true`, `rootNode` will be included in the returned tabbable node array, if `rootNode` is tabbable.

## Differences from jQuery UI's [`:tabbable` selector](https://api.jqueryui.com/tabbable-selector/)

Doesn't need jQuery. Also: doesn't support all the old IE's.

Also: The array accounts for actual tab order.

Also: jQuery UI's `:tabbable` selector ignores elements with height and width of `0`. I'm not sure why — because I've found that I can still tab to those elements. So I kept them in. Only elements hidden with `display: none` or `visibility: hidden` are left out.

Also: This plugin ignores the rarely used `<area>` and `<object>` elements, which are focusable in some circumstances. (If you need them, maybe PR?)

*Feedback more than welcome!*

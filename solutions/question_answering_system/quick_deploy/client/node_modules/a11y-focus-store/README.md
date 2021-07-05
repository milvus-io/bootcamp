# a11y-focus-store

> Accessibility util for storing/restoring focus.

## Installation

```sh
$ npm install --save a11y-focus-store
```

## Usage

```js
import {storeFocus, restoreFocus, clearStoredFocus} from 'a11y-focus-store';

document.body.innerHTML = `
  <button id="button-1">Button 1</button>
  <button id="button-2">Button 2</button>
  <button id="button-3">Button 3</button>
`;

var button1 = document.getElementById('button-1');
var button2 = document.getElementById('button-2');
var button3 = document.getElementById('button-3');

button1.focus();
storeFocus();

button2.focus();
restoreFocus();
// document.activeElement === button1;

button1.focus();
storeFocus();
clearStoredFocus();

innerButton.focus();
// document.activeElement === innerButton;

outerButton.focus();
// document.activeElement === container;

unscopeFocus();
outerButton.focus();
// document.activeElement === outerButton;
```

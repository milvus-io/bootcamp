# Public Methods
To access the component's public methods, add a ref to your typeahead instance and access the ref from a given handler:
```jsx
<div>
  <Typeahead
    ...
    ref={(typeahead) => this.typeahead = typeahead}
  />
  <button onClick={() => this.typeahead.getInstance().clear()}>
    Clear Typeahead
  </button>
</div>
```

Note that you must use `getInstance` to get the typeahead instance.

### `blur()`
Provides a programmatic way to blur the input.

### `clear()`
Provides a programmatic way to reset the component. Calling the method will clear both text and selection(s).

### `focus()`
Provides a programmatic way to focus the input.

### `getInput()`
Provides access to the component's input node.

[Next: Props](Props.md)

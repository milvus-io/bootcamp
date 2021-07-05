import React from 'react';
var DropdownContext = React.createContext({
  menuRef: function menuRef() {},
  toggleRef: function toggleRef() {},
  onToggle: function onToggle() {},
  toggleNode: undefined,
  alignEnd: null,
  show: null,
  drop: null
});
export default DropdownContext;
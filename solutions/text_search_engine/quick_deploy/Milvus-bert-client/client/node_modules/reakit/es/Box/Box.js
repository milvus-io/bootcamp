import { createComponent } from 'reakit-system/createComponent';
import { createHook } from 'reakit-system/createHook';

var useBox = createHook({
  name: "Box",
  keys: ["unstable_system"]
});
var Box = createComponent({
  as: "div",
  useHook: useBox
});

export { Box, useBox };

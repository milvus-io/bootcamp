// @flow

import parse from '../src/parse';
import generate from '../src/generate';

test('basic', () => {
  const parsed = parse({
    color: 'red',
    ':hover': {
      color: 'blue',
    },
    '@media screen': {
      color: 'green',
    },
  });
  const generated = generate(parsed);
  expect(generated).toMatchSnapshot();
});

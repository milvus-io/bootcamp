import parse from '../src/parse';

test('basic', () => {
  const ast = parse({ color: 'red' });
  expect(ast).toMatchSnapshot();
});

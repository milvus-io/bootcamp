/**
 * @jest-environment node
 */

// @flow

/* @jsx css */
import css from '../src';

import React from 'react';
import { renderToString } from 'react-dom/server';
import renderer from 'react-test-renderer';

test('basic', () => {
  const tree = renderer
    .create(
      <div css>
        <div css={{ color: 'red' }}>hello world</div>
        <div css={{ color: 'red' }}>hello world</div>
        <div css={{ color: 'red' }}>hello world</div>
      </div>,
    )
    .toJSON();
  expect(tree).toMatchSnapshot();
});

/**
 * @jest-environment jsdom
 */

// @flow

/* @jsx css */
import css from '../src';
import Glam from '../src/Glam';

import React from 'react';
import renderer from 'react-test-renderer';

const instance = new Glam({ document });

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
  expect(instance.sheet.rules()).toMatchSnapshot();
});

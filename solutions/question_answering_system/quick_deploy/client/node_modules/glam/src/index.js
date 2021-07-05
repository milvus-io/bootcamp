// @flow
import type { Node } from 'react';
import type { RuleGroup } from './types';
import PropTypes from 'prop-types';

import React, { Children, createElement } from 'react';
import { render, hydrate as reactHydrate } from 'react-dom';

import Glam from './Glam';

import parse from './parse';
import generate from './generate';

const isBrowser = typeof window !== 'undefined';

let isHydrating: boolean = false;

const nullClass = parse({}).className;

type Props = {
  css: RuleGroup,
  render: string => Node,
};

type Context = {
  glam: {},
};

// @theme
class Styled extends React.Component<Props> {
  static displayName = 'css';
  static contextTypes = {
    glam: PropTypes.object,
  };
  static childContextTypes = {
    glam: PropTypes.object,
  };
  glam: Glam = this.context.glam ||
    new Glam(isBrowser ? { document } : undefined);
  flush: void => void;
  getChildContext() {
    return {
      glam: this.glam,
    };
  }
  componentWillUnmount() {
    if (this.flush) {
      this.flush();
    }
  }
  render() {
    const { css } = this.props;

    // check parse cache
    // else
    const ast = parse(css);

    const cls = ast.className === nullClass ? '' : ast.className;

    const element = this.props.render(cls);

    if (!isBrowser || (isBrowser && isHydrating)) {
      if (this.glam.isTagged(ast.className)) {
        return element;
      }
      this.glam.tag(ast.className);

      this.flush = () => this.glam.insert(ast); // you already have this content via `$([data-glam='${cls}'])`
      const generated = generate(ast).join('');

      return generated
        ? Children.toArray([
            <style dangerouslySetInnerHTML={{ __html: generated }} />,
            element,
          ])
        : element;
    }
    this.glam.insert(ast);
    return element;
  }
}

export default function glam(
  Type: string,
  props: Object,
  ...children: Array<Node>
) {
  const { css, className, ...rest } = props || {};
  // clean css ?
  if (css) {
    return (
      <Styled
        css={css}
        render={cls => {
          const applyClass = className
            ? cls ? `${className} ${cls}` : className
            : cls;
          return createElement(
            Type,
            applyClass ? { ...rest, className: applyClass } : rest,
            ...children,
          );
        }}
      />
    );
  } else {
    return createElement(Type, props, ...children);
  }
}

export function hydrate(
  element: Node,
  dom: HTMLElement,
  callback: any => void,
) {
  isHydrating = true;
  reactHydrate(element, dom, () => {
    isHydrating = false;
    callback && callback();
  });
}

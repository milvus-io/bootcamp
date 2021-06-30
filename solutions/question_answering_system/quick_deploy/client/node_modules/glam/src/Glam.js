// @flow
import type { AST } from './types';
import StyleSheet from './sheet';

const isBrowser = typeof window !== 'undefined';
import generate from './generate';

type Props = {
  document?: Document,
};

const cache = new WeakMap();

export default class Glam {
  props: Props;
  sheet: ?StyleSheet;
  inserted: {
    [id: string]: boolean | { ast: Object, rules: Array<string> },
  };
  tagged: {
    [id: string]: boolean,
  };
  constructor(props: Props = {}) {
    this.props = props;
    if (isBrowser) {
      const doc = this.props.document;
      const cached = cache.get(doc);
      if (cached) {
        return cached;
      }
      cache.set(doc, this);
      this.sheet = new StyleSheet({ document: doc });
    }

    this.inserted = {};
    this.tagged = {};
  }

  hydrate(ids: Array<string>): void {
    ids.forEach(id => (this.inserted[id] = true));
  }
  tag(id: string): void {
    this.tagged[id] = true;
  }
  isTagged(id: string): boolean {
    return this.tagged[id];
  }

  insert(ast: { className: string, parsed: AST }): void {
    if (!this.inserted[ast.className]) {
      const rules = generate(ast);
      if (isBrowser) {
        rules.forEach(rule => this.sheet.insert(rule));
      }
      this.inserted[ast.className] = true; // on server, add rules instead
    }
  }
}

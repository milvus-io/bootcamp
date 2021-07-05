// @flow
// import assign from 'object-assign'
/*

high performance StyleSheet for css-in-js systems

- uses multiple style tags behind the scenes for millions of rules
- uses `insertRule` for appending in production for *much* faster performance
- 'polyfills' on server side


// usage

import StyleSheet from 'glamor/lib/sheet'
let styleSheet = new StyleSheet()

styleSheet.inject()
- 'injects' the stylesheet into the page (or into memory if on server)

styleSheet.insert('#box { border: 1px solid red; }')
- appends a css rule into the stylesheet


*/

// const doc = global.document;

type Sheet = {
  cssRules: Array<string>,
  insertRule: (string, number) => void,
};

function last(arr) {
  return arr[arr.length - 1];
}

const isBrowser = typeof window !== 'undefined';

const oldIE = ((): boolean => {
  if (isBrowser) {
    let div = document.createElement('div');
    div.innerHTML = '<!--[if lt IE 10]><i></i><![endif]-->';
    return div.getElementsByTagName('i').length === 1;
  }
  return false;
})();

export default class StyleSheet {
  document: Document;
  isSpeedy: boolean;
  maxLength: number;
  ctr: number;
  tags: Array<Object>;
  injected: boolean;
  constructor(
    {
      document,
      speedy = !(process.env.NODE_ENV !== 'production') &&
        !(process.env.NODE_ENV === 'test'),
      maxLength = oldIE ? 4000 : 65000,
    }: { speedy?: boolean, maxLength?: number, document?: Document } = {},
  ) {
    this.document = document;
    this.isSpeedy = speedy; // the big drawback here is that the css won't be editable in devtools
    // this.sheet = undefined;
    this.tags = [];
    this.maxLength = maxLength;
    this.ctr = 0;
    this.inject();
  }
  makeStyleTag() {
    let tag = this.document.createElement('style');
    tag.type = 'text/css';
    tag.setAttribute('data-glamor', '');
    tag.appendChild(this.document.createTextNode(''));
    // todo - use a reference node
    (this.document.head || this.document.getElementsByTagName('head')[0]
    ).appendChild(tag);
    return tag;
  }

  sheetForTag(tag: HTMLElement): ?Sheet {
    if (tag.sheet) {
      return tag.sheet;
    }

    // this weirdness brought to you by firefox
    for (let i = 0; i < this.document.styleSheets.length; i++) {
      if (this.document.styleSheets[i].ownerNode === tag) {
        return this.document.styleSheets[i];
      }
    }
  }
  getSheet(): ?Sheet {
    return this.sheetForTag(last(this.tags));
  }
  inject() {
    if (this.injected) {
      throw new Error('already injected');
    }
    this.tags[0] = this.makeStyleTag();
    this.injected = true;
  }

  _insert(rule: string) {
    // this weirdness for perf, and chrome's weird bug
    // https://stackoverflow.com/questions/20007992/chrome-suddenly-stopped-accepting-insertrule
    try {
      const sheet = this.getSheet();
      sheet &&
        sheet.insertRule(
          rule,
          rule.indexOf('@import') !== -1 ? 0 : sheet.cssRules.length,
        );
    } catch (e) {
      if (process.env.NODE_ENV !== 'production') {
        // might need beter dx for this
        console.warn('whoops, illegal rule inserted', rule); //eslint-disable-line no-console
      }
    }
  }
  insert(rule: string) {
    const sheet = this.getSheet();
    // this is the ultrafast version, works across browsers
    if (this.isSpeedy && sheet && sheet.insertRule) {
      this._insert(rule);
    } else {
      if (rule.indexOf('@import') !== -1) {
        const tag = last(this.tags);
        tag.insertBefore(this.document.createTextNode(rule), tag.firstChild);
      } else {
        last(this.tags).appendChild(this.document.createTextNode(rule));
      }
    }

    this.ctr++;
    if (this.ctr % this.maxLength === 0) {
      this.tags.push(this.makeStyleTag());
    }
  }
  rules() {
    let arr = [];
    this.tags.forEach(tag =>
      arr.splice(arr.length, 0, ...Array.from(this.sheetForTag(tag).cssRules)),
    );
    return arr;
  }
}

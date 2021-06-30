// TypeScript Version: 3.3

import { DomElement } from 'domhandler';
import domToReact from './lib/dom-to-react';
import htmlToDOM from 'html-dom-parser';

export interface HTMLReactParserOptions {
  // TODO: Replace `object` by type for objects like `{ type: 'h1', props: { children: 'Heading' } }`
  replace?: (
    domNode: DomElement
  ) => JSX.Element | object | void | undefined | null | false;
  library?: object;
}

/**
 * Converts HTML string to JSX element(s).
 *
 * @param  html    - The HTML string to parse to JSX element(s).
 * @param  options - The parser options.
 * @return         - Single or array of JSX elements.
 */
declare function HTMLReactParser(
  html: string,
  options?: HTMLReactParserOptions
): ReturnType<typeof domToReact>;

export { DomElement, domToReact, htmlToDOM };

export default HTMLReactParser;

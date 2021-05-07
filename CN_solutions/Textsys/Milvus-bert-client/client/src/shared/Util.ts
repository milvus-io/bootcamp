import { STOP_WORDS } from './Constants';

/* Tokenize words without stopwords and split by punctuation */
export const tokenize = (text: string): Array<string> => {
  let results: Array<string> = [];
  let words = text
    .toLowerCase()
    .replace(/[^\w\s]|_/g, ' ')
    .replace(/\s+/g, ' ')
    .split(' ');

  words.forEach((word) => {
    if (!STOP_WORDS.has(word)) {
      results.push(word);
    }
  });
  return results;
};

export const makePOSTRequest = (url: string, data: Object) => {
  return fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
};

export const makeAsyncPOSTRequest = async (url: string, body: Object) => {
  return await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
};

// Remove "abstract" string from beginning of abstract
export const parseAbstract = (abstract: string): string => {
  return abstract.replace(/^\s*abstract\s*/gi, '');
};

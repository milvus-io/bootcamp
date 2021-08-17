import { OptionTypeBase } from "react-select";

/* Routes */
export const HOME_ROUTE = "/";
export const RELATED_ROUTE = "/related";
export const SEARCH_ROUTE = "/search";

/* Breakpoints */
export const SMALL_MOBILE_BREAKPOINT = 425;
export const LARGE_MOBILE_BREAKPOINT = 600;
export const TABLET_BREAKPOINT = 800;

/* Styles */
export const CONTENT_WIDTH = 1100;

/* API */
export const API_BASE =
  process.env.NODE_ENV === "development"
    ? "http://192.168.1.85:8000/api"
    : "/api";

export const SEARCH_API_BASE =
  process.env.NODE_ENV === "development"
    ? "http://192.168.1.85:5000/text"
    : "/api";

export const SEARCH_PAGE_ENDPOINT = "/v1/search";
export const SEARCH_HOME_PAGE_ENDPOINT = "/search";
export const SEARCH_COLLAPSED_ENDPOINT = "/search/log/collapsed";
export const SEARCH_EXPANDED_ENDPOINT = "/search/log/expanded";
export const SEARCH_CLICKED_ENDPOINT = "/search/log/clicked";
export const RELATED_CLICKED_ENDPOINT = "/related/log/clicked";

export const COUNT = "/count";
export const DROP = "/drop";
export const LOAD = "/load";
export const SEARCH = "/search";

export const RELATED_ENDPOINT = "/related";

/* Search Vertical Models */
export interface SearchVerticalOption extends OptionTypeBase {
  value: string;
  label: string;
}

export const SEARCH_VERTICAL_OPTIONS: Array<SearchVerticalOption> = [
  { value: "cord19", label: "CORD-19" },
  { value: "trialstreamer", label: "Trialstreamer" },
];

/* NLTK Stopwords */
export const STOP_WORDS = new Set([
  "i",
  "me",
  "my",
  "myself",
  "we",
  "our",
  "ours",
  "ourselves",
  "you",
  "your",
  "yours",
  "yourself",
  "yourselves",
  "he",
  "him",
  "his",
  "himself",
  "she",
  "her",
  "hers",
  "herself",
  "it",
  "its",
  "itself",
  "they",
  "them",
  "their",
  "theirs",
  "themselves",
  "what",
  "which",
  "who",
  "whom",
  "this",
  "that",
  "these",
  "those",
  "am",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "having",
  "do",
  "does",
  "did",
  "doing",
  "a",
  "an",
  "the",
  "and",
  "but",
  "if",
  "or",
  "because",
  "as",
  "until",
  "while",
  "of",
  "at",
  "by",
  "for",
  "with",
  "about",
  "against",
  "between",
  "into",
  "through",
  "during",
  "before",
  "after",
  "above",
  "below",
  "to",
  "from",
  "up",
  "down",
  "in",
  "out",
  "on",
  "off",
  "over",
  "under",
  "again",
  "further",
  "then",
  "once",
  "here",
  "there",
  "when",
  "where",
  "why",
  "how",
  "all",
  "any",
  "both",
  "each",
  "few",
  "more",
  "most",
  "other",
  "some",
  "such",
  "no",
  "nor",
  "not",
  "only",
  "own",
  "same",
  "so",
  "than",
  "too",
  "very",
  "s",
  "t",
  "can",
  "will",
  "just",
  "don",
  "should",
  "now",
]);

// @flow
// type Key =
// // [...getComputedStyle(document.body)]

type RuleSet = {
  [key: string]: string | number | null,
};

type Selector = string;
type SelectRuleSet = {
  [key: Selector]: RuleGroup,
};

type MediaQuery = string;
type MediaQueryRuleSet = {
  [key: MediaQuery]: RuleGroup,
};

type Supports = string;
type SupportsRuleSet = {
  [key: Supports]: RuleGroup,
};

export type RuleGroup = RuleSet &
  SelectRuleSet &
  MediaQueryRuleSet &
  SupportsRuleSet;

// type RuleFunction = any => RuleGroup; // styled-system?

export type AST = {
  label: Array<string>,
  plain: RuleSet,
  selects: SelectRuleSet,
  medias: MediaQueryRuleSet,
  supports: SupportsRuleSet,
};

// @flow
function objectToIstf() {}

function IstfToObject() {}

// markers
const RULE_START = 0;
const RULE_END = 1;
const RULE_NAME = 2;
const SELECTOR = 3;
const PARENT_SELECTOR = 4;
const COMPOUND_SELECTOR_START = 5;
const COMPOUND_SELECTOR_END = 6;
const SPACE_COMBINATOR = 5;
const DOUBLED_CHILD_COMBINATOR = 6;
const CHILD_COMBINATOR = 7;
const NEXT_SIBLING_COMBINATOR = 8;
const SUBSEQUENT_SIBLING_COMBINATOR = 9;
const PROPERTY = 10;
const VALUE = 11;
const COMPOUND_VALUE_START = 12;
const COMPOUND_VALUE_END = 13;
const CONDITION = 14;
const FUNCTION_START = 15;
const FUNCTION_END = 16;
const ANIMATION_NAME = 17;
const JS_FUNCTION_SELECTOR = 18;
const JS_FUNCTION_PROPERTY = 19;
const JS_FUNCTION_VALUE = 20;
const JS_FUNCTION_PARTIAL = 21;

// rule types
const STYLE_RULE = 1; // cssom
const CHARSET_RULE = 2;
const IMPORT_RULE = 3; // CSSOM
const MEDIA_RULE = 4; // CSSOM
const FONT_FACE_RULE = 5; // CSSOM
const PAGE_RULE = 6; // CSSOM
const KEYFRAMES_RULE = 7; //css3-animations
const KEYFRAME_RULE = 8; // css3-animations
const MARGIN_RULE = 9; //CSSOM
const NAMESPACE_RULE = 10; // CSSOM
const COUNTER_STYLE_RULE = 11; // css3-lists
const SUPPORTS_RULE = 12; // css3-conditional
const DOCUMENT_RULE = 13; //css3-conditional
const FONT_FEATURE_VALUES_RULE = 14; // css3-fonts
const VIEWPORT_RULE = 15; // css-device-adapt
const REGION_STYLE_RULE = 16; //proposed for css3-regions
const CUSTOM_MEDIA_RULE = 17; // mediaqueries

/*
 * decaffeinate suggestions:
 * DS102: Remove unnecessary code created because of implicit returns
 * DS207: Consider shorter variations of null checks
 * Full docs: https://github.com/decaffeinate/decaffeinate/blob/master/docs/suggestions.md
 */
// Ported from Compass
// https://github.com/Compass/compass/blob/master/core/stylesheets/compass/typography/_units.scss

// Emulate the sass function "unit"
import unit from "./unit"

const baseFontSize = "16px"

// Emulate the sass function "unitless"
const unitLess = length => parseFloat(length)

// Convert any CSS <length> or <percentage> value to any another.
//
// @param length
//   A css <length> value
//
// @param toUnit
//   String matching a css unit keyword, e.g. 'em', 'rem', etc.
//
// @param fromContext
//   When converting from relative units, the absolute length (in px) to
//   which length refers (e.g. for lengths in em units, would normally be the
//   font-size of the current element).
//
// @param toContext
//   For converting to relative units, the absolute length in px to which the
//   output value will refer. Defaults to the same as fromContext, since it is
//   rarely needed.
export default function convertCSSLength(baseFontSize) {
  if (baseFontSize == null) {
    baseFontSize = baseFontSize
  }
  return function(length, toUnit, fromContext, toContext) {
    if (fromContext == null) {
      fromContext = baseFontSize
    }
    if (toContext == null) {
      toContext = fromContext
    }
    const fromUnit = unit(length)

    // Optimize for cases where `from` and `to` units are accidentally the same.
    if (fromUnit === toUnit) {
      return length
    }

    // Convert input length to pixels.
    let pxLength = unitLess(length)

    // Warn if to or from context aren't in pixels.
    // if (unit(fromContext) !== "px") {
    // console.warn(`Parameter fromContext must resolve to a value \
    // in pixel units.`)
    // }
    // if (unit(toContext) !== "px") {
    // console.warn(`Parameter toContext must resolve to a value \
    // in pixel units.`)
    // }

    if (fromUnit !== "px") {
      if (fromUnit === "em") {
        pxLength = unitLess(length) * unitLess(fromContext)
      } else if (fromUnit === "rem") {
        pxLength = unitLess(length) * unitLess(baseFontSize)
      } else if (fromUnit === "ex") {
        pxLength = unitLess(length) * unitLess(fromContext) * 2
      } else {
        return length
      }
      // } else if (["ch", "vw", "vh", "vmin"].includes(fromUnit)) {
      // console.warn(`${fromUnit} units can't be reliably converted; Returning \
      // original value.`)
      // return length
      // } else {
      // console.warn(`${fromUnit} is an unknown or unsupported length unit; \
      // Returning original value.`)
      // return length
      // }
    }

    // Convert length in pixels to the output unit
    let outputLength = pxLength
    if (toUnit !== "px") {
      if (toUnit === "em") {
        outputLength = pxLength / unitLess(toContext)
      } else if (toUnit === "rem") {
        outputLength = pxLength / unitLess(baseFontSize)
      } else if (toUnit === "ex") {
        outputLength = pxLength / unitLess(toContext) / 2
        // } else if (["ch", "vw", "vh", "vmin"].includes(toUnit)) {
        // console.warn(`${toUnit} units can't be reliably converted; Returning \
        // original value.`)
        // return length
        // } else {
        // console.warn(`${toUnit} is an unknown or unsupported length unit; \
        // Returning original value.`)
      } else {
        return length
      }
    }

    return parseFloat(outputLength.toFixed(5)) + toUnit
  }
}

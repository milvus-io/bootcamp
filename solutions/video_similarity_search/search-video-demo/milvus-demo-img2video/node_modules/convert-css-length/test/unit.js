import { expect } from "chai"
import unit from "../src/unit.js"

describe("unit", function() {
  it("should parse unit strings", function() {
    // does not yet support hex or E notation
    expect(unit("20px")).to.equal("px")
    expect(unit("20 gold")).to.equal("gold")
    expect(unit("2.5 px")).to.equal("px")
    expect(unit("2.5 %")).to.equal("%")
    expect(unit("-2.5")).to.equal("")
    expect(unit("0%%")).to.equal("%%")
  })
})

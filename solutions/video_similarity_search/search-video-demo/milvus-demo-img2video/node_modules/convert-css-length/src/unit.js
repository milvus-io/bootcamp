export default function unit(input) {
  return String(input).match(/[\d.\-\+]*\s*(.*)/)[1] || ""
}

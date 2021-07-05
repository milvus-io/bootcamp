export default function flatten(inArr: Array<any>) {
  let arr = [];
  for (let i = 0; i < inArr.length; i++) {
    if (Array.isArray(inArr[i])) arr = arr.concat(flatten(inArr[i]));
    else arr = arr.concat(inArr[i]);
  }
  return arr;
}

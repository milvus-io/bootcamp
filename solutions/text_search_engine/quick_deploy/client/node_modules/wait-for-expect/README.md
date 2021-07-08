[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![CircleCI](https://circleci.com/gh/TheBrainFamily/wait-for-expect.svg?style=shield)](https://circleci.com/gh/TheBrainFamily/wait-for-expect)

# wait-for-expect
Wait for expectation to be true, useful for integration and end to end testing

Think things like calling external APIs, database operations, or even GraphQL subscriptions. 
We will add examples for all of them soon, for now please enjoy the simple docs. :-)

# Usage:

```javascript
const waitForExpect = require("wait-for-expect")

test("it waits for the number to change", async () => {
  let numberToChange = 10;
  // we are using random timeout here to simulate a real-time example
  // of an async operation calling a callback at a non-deterministic time
  const randomTimeout = Math.floor(Math.random() * 300);

  setTimeout(() => {
    numberToChange = 100;
  }, randomTimeout);

  await waitForExpect(() => {
    expect(numberToChange).toEqual(100);
  });
});
```

instead of:

```javascript

test("it waits for the number to change", () => {
  let numberToChange = 10;
  const randomTimeout = Math.floor(Math.random() * 300);

  setTimeout(() => {
    numberToChange = 100;
  }, randomTimeout);
  
  setTimeout(() => {
    expect(numberToChange).toEqual(100);
  }, 700);
});
```

It will check whether the expectation passes right away in the next available "tick" (very useful with, for example, integration testing of react when mocking fetches, like here: https://github.com/kentcdodds/react-testing-library#usage).

If it doesn't, it will keep repeating for the duration of, at most, the specified timeout, every 50 ms. The default timeout is 4.5 seconds to fit below the default 5 seconds that Jest waits for before throwing an error. 

Nice thing about this simple tool is that if the expectation keeps failing till the timeout, it will check it one last time, but this time the same way your test runner would run it - so you basically get your expectation library error, the sam way like if you used setTimeout to wait but didn't wait long enough.

To show an example - if I change the expectation to wait for 105 in above code, you will get nice and familiar:

```

 FAIL  src/waitForExpect.spec.js (5.042s)
  ✕ it waits for the number to change (4511ms)

  ● it waits for the number to change

    expect(received).toEqual(expected)
    
    Expected value to equal:
      105
    Received:
      100

       9 |   }, 600);
      10 |   await waitForExpect(() => {
    > 11 |     expect(numberToChange).toEqual(105);
      12 |   });
      13 | });
      14 | 
      
      at waitForExpect (src/waitForExpect.spec.js:11:28)
      at waitUntil.catch (src/index.js:61:5)

Test Suites: 1 failed, 1 total
Tests:       1 failed, 1 total
Snapshots:   0 total
Time:        5.807s
```

You can add multiple expectations to wait for, all of them have to pass, and if one of them don't, it will be marked.
For example, let's add another expectation for a different number, notice how jest tells you that that's the expectation that failed.

```
    expect(received).toEqual(expected)
    
    Expected value to equal:
      110
    Received:
      105

      11 |   await waitForExpect(() => {
      12 |     expect(numberToChange).toEqual(100);
    > 13 |     expect(numberThatWontChange).toEqual(110);
      14 |   });
      15 | });
      16 | 
      
      at waitForExpect (src/waitForExpect.spec.js:13:34)
      at waitUntil.catch (src/index.js:61:5)
```

Since 0.6.0 we can now work with promises, for example, this is now possible:

```javascript
test("rename todo by typing", async () => {
  // (..)
  const todoToChange = getTodoByText("original todo");
  todoToChange.value = "different text now";
  Simulate.change(todoToChange);

  await waitForExpect(() =>
    expect(
      todoItemsCollection.findOne({
        text: "different text now"
      })).resolves.not.toBeNull()
  );
});
```

Async Await also works, as in this example - straight from our test case

```javascript
test("it works with promises", async () => {
  let numberToChange = 10;
  const randomTimeout = Math.floor(Math.random() * 300);

  setTimeout(() => {
    numberToChange = 100;
  }, randomTimeout);

  const sleep = (ms) =>
    new Promise(resolve => setTimeout(() => resolve(), ms));

  await waitForExpect(async () => {
    await sleep(10);
    expect(numberToChange).toEqual(100);
  });
});
```

(Note: Obviously, in this case it doesn't make sense to put the await sleep there, this is just for demonstration purpose)

# API
waitForExpect takes 3 arguments, 2 optional.

```javascript
/**
 * Waits for predicate to not throw and returns a Promise
 *
 * @param  expectation  Function  Predicate that has to complete without throwing
 * @param  timeout  Number  Maximum wait interval, 4500ms by default
 * @param  interval  Number  Wait interval, 50ms by default
 * @return  Promise  Promise to return a callback result
 */
```

The defaults for `timeout` and `interval` can also be edited globally, e.g. in a jest setup file:
```javascript
import waitForExpect from 'wait-for-expect';

waitForExpect.defaults.timeout = 2000;
waitForExpect.defaults.interval = 10;
```

## Changelog
1.0.0 - 15 June 2018

( For most people this change doesn't matter. )
Export the function directly in module.exports instead of exporting as an object that has default key. If that's not clear (...it isn't ;-) ) - check #8 #9 . 
Thanks to @mbaranovski for the PR and @BenBrostoff for creating the issue! I'm making this 1.0.0 as this is breaking for people that currently did:
```javascript
const { default: waitFor } = require('wait-for-expect');
```

0.6.0 - 3 May 2018

Work with promises.

0.5.0 - 10 April 2018

Play nicely with jest fake timers (and also in any test tool that overwrites setTimeout) - thanks to @slightlytyler and @kentcoddods for helping to get this resolved.

## Credit
Originally based on ideas from https://github.com/devlato/waitUntil.
Simplified highly and rewritten for 0.1.0 version.
Simplified even more and rewritten even more for 0.2.0 with guidance from Kent C. Dodds: https://github.com/kentcdodds/react-testing-library/pull/25
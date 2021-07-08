import React from "react";
import { render, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/extend-expect";
import userEvent from "../../src";

afterEach(cleanup);

describe("userEvent.dblClick", () => {
  it.each(["input", "textarea"])(
    "should fire the correct events for <%s>",
    type => {
      const events = [];
      const eventsHandler = jest.fn(evt => events.push(evt.type));
      const { getByTestId } = render(
        React.createElement(type, {
          "data-testid": "element",
          onMouseOver: eventsHandler,
          onMouseMove: eventsHandler,
          onMouseDown: eventsHandler,
          onFocus: eventsHandler,
          onMouseUp: eventsHandler,
          onClick: eventsHandler,
          onDoubleClick: eventsHandler
        })
      );

      userEvent.dblClick(getByTestId("element"));

      expect(events).toEqual([
        "mouseover",
        "mousemove",
        "mousedown",
        "focus",
        "mouseup",
        "click",
        "mousedown",
        "mouseup",
        "click",
        "dblclick"
      ]);
    }
  );

  it('should fire the correct events for <input type="checkbox">', () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render(
      <input
        data-testid="element"
        type="checkbox"
        onMouseOver={eventsHandler}
        onMouseMove={eventsHandler}
        onMouseDown={eventsHandler}
        onFocus={eventsHandler}
        onMouseUp={eventsHandler}
        onClick={eventsHandler}
        onChange={eventsHandler}
      />
    );

    userEvent.dblClick(getByTestId("element"));

    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click",
      "change",
      "mousedown",
      "mouseup",
      "click",
      "change"
    ]);

    expect(getByTestId("element")).toHaveProperty("checked", false);
  });

  it("should fire the correct events for <div>", () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render(
      <div
        data-testid="div"
        onMouseOver={eventsHandler}
        onMouseMove={eventsHandler}
        onMouseDown={eventsHandler}
        onFocus={eventsHandler}
        onMouseUp={eventsHandler}
        onClick={eventsHandler}
      />
    );

    userEvent.dblClick(getByTestId("div"));
    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click",
      "mousedown",
      "mouseup",
      "click"
    ]);
  });
});

import React from "react";
import { render, cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/extend-expect";
import userEvent from "../../src";

afterEach(cleanup);

describe("userEvent.click", () => {
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
          onClick: eventsHandler
        })
      );

      userEvent.click(getByTestId("element"));

      expect(events).toEqual([
        "mouseover",
        "mousemove",
        "mousedown",
        "focus",
        "mouseup",
        "click"
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

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click",
      "change"
    ]);

    expect(getByTestId("element")).toHaveProperty("checked", true);
  });

  it('should fire the correct events for <input type="checkbox" disabled>', () => {
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
        disabled
      />
    );

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([]);

    expect(getByTestId("element")).toHaveProperty("checked", false);
  });

  it('should fire the correct events for <input type="radio">', () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render(
      <input
        data-testid="element"
        type="radio"
        onMouseOver={eventsHandler}
        onMouseMove={eventsHandler}
        onMouseDown={eventsHandler}
        onFocus={eventsHandler}
        onMouseUp={eventsHandler}
        onClick={eventsHandler}
        onChange={eventsHandler}
      />
    );

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click",
      "change"
    ]);

    expect(getByTestId("element")).toHaveProperty("checked", true);
  });

  it('should fire the correct events for <input type="radio" disabled>', () => {
    const events = [];
    const eventsHandler = jest.fn(evt => events.push(evt.type));
    const { getByTestId } = render(
      <input
        data-testid="element"
        type="radio"
        onMouseOver={eventsHandler}
        onMouseMove={eventsHandler}
        onMouseDown={eventsHandler}
        onFocus={eventsHandler}
        onMouseUp={eventsHandler}
        onClick={eventsHandler}
        onChange={eventsHandler}
        disabled
      />
    );

    userEvent.click(getByTestId("element"));

    expect(events).toEqual([]);

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

    userEvent.click(getByTestId("div"));
    expect(events).toEqual([
      "mouseover",
      "mousemove",
      "mousedown",
      "mouseup",
      "click"
    ]);
  });

  it("toggles the focus", () => {
    const { getByTestId } = render(
      <React.Fragment>
        <input data-testid="A" />
        <input data-testid="B" />
      </React.Fragment>
    );

    const a = getByTestId("A");
    const b = getByTestId("B");

    expect(a).not.toHaveFocus();
    expect(b).not.toHaveFocus();

    userEvent.click(a);
    expect(a).toHaveFocus();
    expect(b).not.toHaveFocus();

    userEvent.click(b);
    expect(a).not.toHaveFocus();
    expect(b).toHaveFocus();
  });

  it.each(["input", "textarea"])(
    "gives focus to <%s> when clicking a <label> with htmlFor",
    type => {
      const { getByTestId } = render(
        <React.Fragment>
          <label htmlFor="input" data-testid="label">
            Label
          </label>
          {React.createElement(type, { id: "input", "data-testid": "input" })}
        </React.Fragment>
      );
      userEvent.click(getByTestId("label"));
      expect(getByTestId("input")).toHaveFocus();
    }
  );

  it.each(["input", "textarea"])(
    "gives focus to <%s> when clicking a <label> without htmlFor",
    type => {
      const { getByTestId } = render(
        <React.Fragment>
          <label data-testid="label">
            Label
            {React.createElement(type, { "data-testid": "input" })}
          </label>
        </React.Fragment>
      );
      userEvent.click(getByTestId("label"));
      expect(getByTestId("input")).toHaveFocus();
    }
  );

  it.each(["input", "textarea"])(
    "gives focus to <%s> when clicking on an element contained within a <label>",
    type => {
      const { getByText, getByTestId } = render(
        <React.Fragment>
          <label htmlFor="input" data-testid="label">
            <span>Label</span>
          </label>
          {React.createElement(type, { id: "input", "data-testid": "input" })}
        </React.Fragment>
      );
      userEvent.click(getByText("Label"));
      expect(getByTestId("input")).toHaveFocus();
    }
  );

  it('checks <input type="checkbox"> when clicking a <label> with htmlFor', () => {
    const { getByTestId } = render(
      <React.Fragment>
        <label htmlFor="input" data-testid="label">
          Label
        </label>
        <input id="input" data-testid="input" type="checkbox" />
      </React.Fragment>
    );
    expect(getByTestId("input")).toHaveProperty("checked", false);
    userEvent.click(getByTestId("label"));
    expect(getByTestId("input")).toHaveProperty("checked", true);
  });

  it('checks <input type="checkbox"> when clicking a <label> without htmlFor', () => {
    const { getByTestId } = render(
      <React.Fragment>
        <label data-testid="label">
          Label
          <input id="input" data-testid="input" type="checkbox" />
        </label>
      </React.Fragment>
    );
    expect(getByTestId("input")).toHaveProperty("checked", false);
    userEvent.click(getByTestId("label"));
    expect(getByTestId("input")).toHaveProperty("checked", true);
  });

  it("should submit a form when clicking on a <button>", () => {
    const onSubmit = jest.fn();
    const { getByText } = render(
      <form onSubmit={onSubmit}>
        <button>Submit</button>
      </form>
    );
    userEvent.click(getByText("Submit"));
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it('should not submit a form when clicking on a <button type="button">', () => {
    const onSubmit = jest.fn();
    const { getByText } = render(
      <form onSubmit={onSubmit}>
        <button type="button">Submit</button>
      </form>
    );
    userEvent.click(getByText("Submit"));
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

import { render, cleanup } from "@testing-library/vue";
import "@testing-library/jest-dom/extend-expect";
import userEvent from "../../src";

afterEach(cleanup);

describe("userEvent.dblClick", () => {
  it.each(["input", "textarea"])(
    "should fire the correct events for <%s>",
    type => {
      const events = [];
      const eventsHandler = jest.fn(evt => events.push(evt.type));
      const { getByTestId } = render({
        render: function(h) {
          return h(type, {
            attrs: {
              "data-testid": "element"
            },
            on: {
              mouseover: eventsHandler,
              mousemove: eventsHandler,
              mousedown: eventsHandler,
              focus: eventsHandler,
              mouseup: eventsHandler,
              click: eventsHandler,
              dblclick: eventsHandler
            }
          });
        }
      });

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

    const { getByTestId } = render({
      render: function(h) {
        return h("input", {
          attrs: {
            type: "checkbox",
            "data-testid": "element"
          },
          on: {
            mouseover: eventsHandler,
            mousemove: eventsHandler,
            mousedown: eventsHandler,
            focus: eventsHandler,
            mouseup: eventsHandler,
            click: eventsHandler,
            change: eventsHandler
          }
        });
      }
    });

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
    const { getByTestId } = render({
      render: function(h) {
        return h("div", {
          attrs: {
            "data-testid": "div"
          },
          on: {
            mouseover: eventsHandler,
            mousemove: eventsHandler,
            mousedown: eventsHandler,
            focus: eventsHandler,
            mouseup: eventsHandler,
            click: eventsHandler,
            change: eventsHandler
          }
        });
      }
    });

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

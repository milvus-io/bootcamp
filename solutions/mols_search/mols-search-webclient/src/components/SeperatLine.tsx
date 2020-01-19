import React from "react";
import { baseColor } from "../utils/color";

const SeperatLine = (props: any) => {
  const { title, end = "", onEndClick, style = {} } = props;
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        color: "rgba(176,176,185,0.75)",
        marginBottom: "10px",
        ...style
      }}
    >
      <p style={{ marginRight: "20px" }}>{title}</p>
      <div
        style={{
          flexGrow: 1,
          textAlign: "center",
          minWidth: "200px",
          display: "flex",
          alignItems: "center",
          fontSize: "15px"
        }}
      >
        <hr
          style={{
            width: "100%",
            border: "none",
            height: "1px",
            backgroundColor: "rgba(176,176,185,0.75)",
            marginRight: "20px"
          }}
        />
      </div>
      <p onClick={onEndClick} style={{ color: baseColor, cursor: "pointer" }}>
        {end}
      </p>
    </div>
  );
};

export default SeperatLine;

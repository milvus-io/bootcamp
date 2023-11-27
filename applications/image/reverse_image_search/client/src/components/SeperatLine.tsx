import React from "react";

const SeperatLine = (props: any) => {
  const { title, style={} } = props;
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        color: "rgba(176,176,185,0.75)",
        marginBottom:"10px",
        ...style,
      }}
    >
      <p style={{ marginRight: "20px"}}>{title}</p>
      <div
        style={{
          flexGrow: 1,
          textAlign: "center",
          display: "flex",
          alignItems: "center",
          fontSize:'15px'
        }}
      >
        <hr
          style={{
            width: "100%",
            border: "none",
            height: "1px",
            backgroundColor: "rgba(176,176,185,0.75)"
          }}
        />
      </div>
    </div>
  );
};

export default SeperatLine;

import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCrop } from "@fortawesome/free-solid-svg-icons";

const ImageResizeButton = ({ onClick, resizing, imageResizeModel }) =>
  imageResizeModel ? (
    <button className="button is-primary" onClick={onClick}>
      <span className="icon">
        <FontAwesomeIcon icon={faCrop} />
      </span>
      &nbsp; {resizing ? "Resizing..." : "Resize"}
    </button>
  ) : (
    <div></div>
  );

export default ImageResizeButton;

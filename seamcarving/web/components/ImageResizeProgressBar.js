import React from "react";

const ImageResizeProgressBar = ({ resizing }) =>
  resizing ? (
    <progress className="progress is-primary" max="100">
      40%
    </progress>
  ) : (
    <div></div>
  );

export default ImageResizeProgressBar;

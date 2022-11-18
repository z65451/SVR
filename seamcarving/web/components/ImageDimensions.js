import React from "react";

const ImageDimensions = ({ dimensions, crop }) => {
  const currentDimensions = dimensions ? (
    <span>
      Current dimensions:
      <strong>
        ({dimensions.width}, {dimensions.height})
      </strong>
    </span>
  ) : (
    <div></div>
  );
  const cropDimensions = crop ? (
    <span>
      New dimensions:
      <strong>
        ({crop.width}, {crop.height})
      </strong>
    </span>
  ) : (
    <div></div>
  );

  return (
    <div className="field is-grouped is-grouped-centered">
      <div className="control">
        <div className="control is-fullwidth">{currentDimensions}</div>
        <div className="control is-fullwidth">{cropDimensions}</div>
      </div>
    </div>
  );
};

export default ImageDimensions;

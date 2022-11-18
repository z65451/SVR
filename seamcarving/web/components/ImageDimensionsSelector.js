import React from "react";
import ReactCrop from "react-image-crop";

const ImageDimensionsSelector = ({
  name,
  file,
  crop,
  onImageLoaded,
  onCropSelectionChanged,
  onCropSelectionCompleted,
}) => {
  return (
    <form>
      <div className="field">
        <div className="control">
          <h2>{name}</h2>
        </div>
      </div>
      <div className="field">
        <div className="control">
          {file && (
            <ReactCrop
              src={file}
              crop={crop}
              ruleOfThirds
              onImageLoaded={onImageLoaded}
              onComplete={onCropSelectionCompleted}
              onChange={onCropSelectionChanged}
            />
          )}
        </div>
      </div>
    </form>
  );
};

export default ImageDimensionsSelector;

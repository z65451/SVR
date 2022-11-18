import React from "react";

import ImageSelector from "../ImageSelector";
import CroppedImagePreview from "../CroppedImagePreview";
import ImageDimensions from "../ImageDimensions";
import ImageResizeButton from "../ImageResizeButton";
import ImageResizeProgressBar from "../ImageResizeProgressBar";

const ImageResizingSelectionArea = ({
  state,
  onImageResizeStart,
  onImageSelected,
  onDimensionsSelected,
}) => {
  const {
    resizeResultImages,
    resizing,
    imageResizeModel,
    dimensions,
    crop,
    imageName,
  } = state;

  const croppedImageUrl = resizeResultImages
    ? resizeResultImages.cropped_url
    : null;

  return (
    <article className="tile is-child box">
      <div className="is-flex is-horizontal-center">
        <br />
        <div>
          <ImageDimensions crop={crop} dimensions={dimensions} />
          <div className="field">
            <div className="control">
              <ImageResizeButton
                onClick={onImageResizeStart}
                resizing={resizing}
                imageResizeModel={imageResizeModel}
              />
            </div>
          </div>
          <div className="field">
            <ImageSelector
              croppedImageUrl={croppedImageUrl}
              onImageSelected={onImageSelected}
              onDimensionsSelected={onDimensionsSelected}
            />
            <CroppedImagePreview
              croppedImageUrl={croppedImageUrl}
              croppedImageName={imageName}
            />
          </div>
          <div className="field">
            <div className="control">
              <ImageResizeProgressBar resizing={resizing} />
            </div>
          </div>
        </div>
      </div>
    </article>
  );
};

export default ImageResizingSelectionArea;

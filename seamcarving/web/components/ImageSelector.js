import React from "react";

import ImageFilePicker from "./ImageFilePicker";
import ImageDimensionsSelector from "./ImageDimensionsSelector";
import ImageResizeService from "../services/ImageResizeService";

export default class ImagesSelector extends React.PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      src: { file: null, name: null },
      crop: {
        unit: "%",
        width: 50,
        height: 50,
      },
    };
    this.imageResizeService = new ImageResizeService();
  }

  onImageSelected = (e) => {
    const { files } = e.target;

    if (files && files.length > 0) {
      const reader = new FileReader();
      const [selectedFile, ...rest] = files;
      reader.addEventListener("load", () =>
        this.setState({
          src: { file: reader.result, name: selectedFile.name },
        })
      );
      reader.readAsDataURL(selectedFile);
      this.setState({ src: { name: selectedFile.name } });
      this.props.onImageSelected(this.state.src);
    }
  };

  onImageLoaded = (image) => {
    this.imageRef = image;
    this.setState({
      dimensions: {
        height: image.offsetHeight,
        width: image.offsetWidth,
      },
    });
  };

  onCropSelectionCompleted = (crop) => {
    const { onDimensionsSelected } = this.props;
    const { dimensions, src } = this.state;
    onDimensionsSelected({
      model: this.buildImageResizeRequest(crop),
      crop,
      dimensions,
      imageName: src.name,
    });
  };

  onCropSelectionChanged = (crop) => {
    this.setState({ crop });
  };

  buildImageResizeRequest(crop) {
    const { src, dimensions } = this.state;
    const { file } = src;

    if (!dimensions) return undefined;

    if (this.imageRef && crop.width && crop.height) {
      return this.imageResizeService.buildImageResizeModel({
        file,
        dimensions,
        crop,
      });
    }
    return undefined;
  }

  render() {
    const { crop, src } = this.state;
    const { name, file } = src;

    return (
      <div>
        <ImageFilePicker
          imageName={name}
          onImageSelected={this.onImageSelected}
        />
        <br />
        <div className="column">
          <ImageDimensionsSelector
            name={name}
            file={file}
            crop={crop}
            onImageLoaded={this.onImageLoaded}
            onCropSelectionChanged={this.onCropSelectionChanged}
            onCropSelectionCompleted={this.onCropSelectionCompleted}
            onImageSelected={this.onImageSelected}
          />
        </div>
      </div>
    );
  }
}

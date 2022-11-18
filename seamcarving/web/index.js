import React from "react";
import ReactDOM from "react-dom";
import "regenerator-runtime/runtime";

import Header from "./components/layout/Header";
import ImageResizeService from "./services/ImageResizeService";
import ImageResizingResults from "./components/layout/ImageResizingResults";
import ImageResizingSelectionArea from "./components/layout/ImageResizingSelectionArea";
import "./services/ImageResizeService";

import "bulma/css/bulma.css";
import "./styles.css";
import "react-image-crop/dist/ReactCrop.css";

class AppComponent extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      imageName: null,
      resizeResultImages: undefined,
      resized: undefined,
      resizing: undefined,
      imageResizeModel: null,
      crop: null,
      dimensions: null,
    };
    this.imageResizeService = new ImageResizeService();
  }

  resetState() {
    this.setState({
      imageName: null,
      resizeResultImages: undefined,
      resized: undefined,
      resizing: undefined,
      imageResizeModel: null,
      crop: null,
      dimensions: null,
    });
  }

  resizeImage() {
    const { imageResizeModel } = this.state;

    if (!imageResizeModel) return;

    this.setState({ resizing: true, resized: false });

    this.imageResizeService
      .resizeImage(imageResizeModel)
      .then((resultImages) => {
        this.setState({
          resizeResultImages: resultImages,
          resizing: false,
          resized: true,
        });
      });
  }

  onDimensionsSelected = ({ model, crop, dimensions, imageName }) => {
    this.setState({
      imageResizeModel: model,
      crop,
      dimensions,
      imageName,
    });
  };

  onImageSelected = () => {
    this.resetState();
  };

  onImageResizeStart = () => {
    this.resizeImage();
  };

  render() {
    const { resized, resizeResultImages } = this.state;

    return (
      <div>
        <Header />
        <br />
        <div className="container">
          <div className="tile is-ancestor has-text-centered">
            <ImageResizingSelectionArea
              state={this.state}
              onImageResizeStart={this.onImageResizeStart}
              onImageSelected={this.onImageSelected}
              onDimensionsSelected={this.onDimensionsSelected}
            />
          </div>
          <div>
            <ImageResizingResults
              resized={resized}
              resizeResultImages={resizeResultImages}
            />
          </div>
        </div>
      </div>
    );
  }
}

const App = document.getElementById("app");

ReactDOM.render(<AppComponent name="App" />, App);

import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faFileUpload } from "@fortawesome/free-solid-svg-icons";

const ImageFilePicker = ({ imageName, onImageSelected }) => {
  return (
    <div className="is-flex is-horizontal-center">
      <div className="file has-name">
        <label className="file-label">
          <input
            className="file-input"
            type="file"
            accept="image/*"
            onChange={onImageSelected}
          ></input>
          <span className="file-cta">
            <span className="file-icon">
              <i className="fas fa-upload"></i>
            </span>
            <span className="file-label">
              <span className="icon">
                <FontAwesomeIcon icon={faFileUpload} />
              </span>
              &nbsp; Choose an image…
            </span>
          </span>
          <span className="file-name">{imageName}</span>
        </label>
      </div>
    </div>
  );
};

export default ImageFilePicker;

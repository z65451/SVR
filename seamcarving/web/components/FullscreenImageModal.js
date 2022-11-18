import React, { Component } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRight } from "@fortawesome/free-solid-svg-icons";

export default class FullscreenImageModal extends Component {
  constructor(props) {
    super(props);
    this.state = {
      active: false,
    };
  }

  changeModalActivity = ({ active }) => {
    this.setState({ active });
  };

  render() {
    const { active } = this.state;
    const { imageUrl } = this.props;
    const modalClass = active ? "modal is-active" : "modal";

    return (
      <div>
        <div className="field is-expanded">
          <button
            className="button is-light is-fullwidth"
            onClick={() => {
              this.changeModalActivity({ active: true });
            }}
          >
            <span className="icon">
              <FontAwesomeIcon icon={faArrowRight} />
            </span>
            &nbsp; view fullscreen
          </button>
        </div>
        <div className={modalClass}>
          <div className="modal-background"></div>
          <div className="modal-content">
            <p className="image is-4by3">
              <img src={imageUrl}></img>
            </p>
          </div>
          <button
            className="modal-close is-large"
            aria-label="close"
            onClick={() => {
              this.changeModalActivity({ active: false });
            }}
          ></button>
        </div>
      </div>
    );
  }
}
